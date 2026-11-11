// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// graft-kext's command line: build a consumer kext for Graft to bind, or check
// one in a prelinked collection. usage() below is its reference.
#include "consumer.h"
#include "prelink.h"
#include "process.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace graft::tool {
namespace {

namespace fs = std::filesystem;

const char Usage[] =
        "usage: graft-kext build --name=NAME --version=VERSION --arch=ARCH\n"
        "                        --declaration=DECLARATION --info-plist=INFO_PLIST\n"
        "                        --work=WORK --output=KEXT OBJECT... [-- COMPILER_FLAGS...]\n"
        "       graft-kext probe DECLARATION LAYOUTS [-- COMPILER_FLAGS...]\n"
        "       graft-kext reserve BINARY DECLARATION LAYOUTS INFO_PLIST RESERVATION\n"
        "       graft-kext stage KEXT DECLARATION LAYOUTS OUTPUT\n"
        "       graft-kext check-prelinked COLLECTION KEXT...\n";

const char Description[] =
        "\n"
        "Build a consumer kext for Graft to bind, or check one in a prelinked collection.\n"
        "\n"
        "DECLARATION is the consumer's graft.json. build, which consumer.mk runs, does every\n"
        "step of a consumer's build from its objects, leaving each step's output in WORK:\n"
        "\n"
        "- probe, into WORK/layouts.json, with COMPILER_FLAGS for ARCH's architecture;\n"
        "- the first link of the OBJECTs, with ARCH, into WORK/census;\n"
        "- the Info.plist with @VERSION@ replaced by VERSION, checked with plutil -lint,\n"
        "  into WORK/Unbound.kext;\n"
        "- reserve, into WORK/manifest-reserve.bin;\n"
        "- the second link, with that reservation, into WORK/Unbound.kext as NAME;\n"
        "- stage, from WORK/Unbound.kext into KEXT, which it replaces.\n"
        "\n"
        "ARCH is the links' target flags as one argument, such as \"-arch x86_64\". It\n"
        "names one architecture, and COMPILER_FLAGS name no other. The other commands are\n"
        "those steps alone:\n"
        "\n"
        "probe compiles a subclass of each declared class with the consumer's compiler\n"
        "flags and writes LAYOUTS: each class's size and the method in each of its vtable\n"
        "slots.\n"
        "\n"
        "reserve reads BINARY, the consumer linked without a manifest, and writes\n"
        "RESERVATION, a zero-filled file as large as the manifest and its import\n"
        "placeholders. The consumer is then linked again with -sectcreate __DATA __graft\n"
        "RESERVATION.\n"
        "\n"
        "stage copies KEXT, the bundle around that second link, to OUTPUT, writing the\n"
        "manifest into it and leaving the consumer dormant for Graft to bind.\n"
        "\n"
        "check-prelinked checks that COLLECTION, an x86-64 kernel collection OpenCore\n"
        "prelinked Graft and the kexts it serves into, keeps each staged consumer KEXT\n"
        "bindable.\n"
        "\n"
        "A command that refuses its input prints the reason after \"consumer:\" (or\n"
        "\"prelink:\" for check-prelinked) and exits 1; a malformed command line exits 2.\n";

// A malformed command line, which prints the usage and exits 2.
struct UsageError {
    std::string reason;
};

// path as Python's pathlib writes it: without empty or "." parts and without a
// trailing slash, so that the paths in a tool's command line read the same.
std::string normalized(const std::string &path) {
    std::string out;
    size_t at = 0;
    while (at <= path.size()) {
        size_t end = path.find('/', at);
        if (end == std::string::npos)
            end = path.size();
        std::string part = path.substr(at, end - at);
        if (!part.empty() && part != ".")
            out += (out.empty() ? "" : "/") + part;
        at = end + 1;
    }
    if (!path.empty() && path[0] == '/')
        return "/" + out;
    return out.empty() ? "." : out;
}

std::string join(const std::string &directory, const std::string &name) {
    return directory == "/" ? "/" + name : directory + "/" + name;
}

// Run a build tool, which reports its own errors, and refuse the build if it fails.
// quiet discards what the tool prints when it succeeds, such as plutil -lint's OK.
void tool(const std::vector<std::string> &command, bool quiet = false) {
    if (!run(command, quiet ? "/dev/null" : ""))
        throw Error(shellJoin(command) + " failed");
}

// Link objects as a kext into output, with reservation as its __DATA,__graft when
// one is given.
void link(const std::vector<std::string> &arch, const std::vector<std::string> &objects,
          const std::string &output, const std::string &reservation = "") {
    std::vector<std::string> command = {"xcrun", "clang++"};
    command.insert(command.end(), arch.begin(), arch.end());
    command.insert(command.end(), {"-nostdlib", "-Xlinker", "-kext"});
    // An arm64e consumer keeps its import slots and vtables in writable __DATA,
    // where Graft writes them after the kernel has locked __DATA_CONST, so its
    // link folds __DATA_CONST into __DATA.
    for (const std::string &flag : arch) {
        if (flag.find("arm64e") != std::string::npos) {
            command.insert(command.end(), {"-Xlinker", "-no_data_const"});
            break;
        }
    }
    command.insert(command.end(), objects.begin(), objects.end());
    command.push_back("-lcc_kext");
    if (!reservation.empty())
        command.insert(command.end(), {"-Xlinker", "-sectcreate", "-Xlinker", "__DATA", "-Xlinker",
                                       "__graft", "-Xlinker", reservation});
    command.insert(command.end(), {"-o", output});
    tool(command);
}

void probe(const std::string &declaration, const std::string &layouts,
           const std::vector<std::string> &flags) {
    Declaration config = readDeclaration(declaration);
    writeText(layouts, layoutsText(probeLayouts(config, flags)));
}

void reserve(const std::string &binary, const std::string &declaration, const std::string &layouts,
             const std::string &infoPlist, const std::string &reservation) {
    Declaration config = readDeclaration(declaration);
    Layouts measured = readLayouts(layouts);
    Activation activation = consumerActivation(parsePlist(readFile(infoPlist), infoPlist));
    uint64_t size = reservationSize(Image(readFile(binary)), config, measured, activation);
    writeFile(reservation, Bytes(size, 0));
}

void stage(const std::string &kext, const std::string &declaration, const std::string &layouts,
           const std::string &output) {
    Declaration config = readDeclaration(declaration);
    stageBundle(kext, config, readLayouts(layouts), output);
}

struct BuildOptions {
    std::string name, version, declaration, infoPlist, work, output;
    std::vector<std::string> arch, objects, flags;
};

// The flags the probe compiles with: COMPILER_FLAGS, for the one architecture
// ARCH names. The probe lays the classes out for that architecture, which the
// links then target, so flags naming another are refused rather than measured.
std::vector<std::string> probeFlags(const BuildOptions &options) {
    auto architectures = [](const std::vector<std::string> &flags) {
        std::vector<std::string> named;
        for (size_t i = 0; i + 1 < flags.size(); ++i) {
            if (flags[i] == "-arch")
                named.push_back(flags[i + 1]);
        }
        return named;
    };
    std::vector<std::string> target = architectures(options.arch);
    if (target.size() != 1)
        throw Error("ARCH must name one architecture, as -arch x86_64 does");
    for (const std::string &named : architectures(options.flags)) {
        if (named != target[0])
            throw Error("the compiler flags target " + named + ", not ARCH's " + target[0]);
    }
    std::vector<std::string> flags = {"-arch", target[0]};
    flags.insert(flags.end(), options.flags.begin(), options.flags.end());
    return flags;
}

void build(const BuildOptions &options) {
    std::string layouts = join(options.work, "layouts.json");
    std::string census = join(options.work, "census");
    std::string reservation = join(options.work, "manifest-reserve.bin");
    std::string unbound = join(options.work, "Unbound.kext");
    std::string versioned = join(unbound, "Contents/Info.plist");
    std::vector<std::string> flags = probeFlags(options);
    std::error_code error;
    fs::create_directories(options.work, error);
    probe(options.declaration, layouts, flags);
    link(options.arch, options.objects, census);
    // Staging copies the whole unbound bundle, so it starts empty each time.
    fs::remove_all(unbound, error);
    fs::create_directories(join(unbound, "Contents"), error);
    std::string info = readText(options.infoPlist);
    for (size_t at = info.find("@VERSION@"); at != std::string::npos;
         at = info.find("@VERSION@", at + options.version.size())) {
        info.replace(at, 9, options.version);
    }
    writeText(versioned, info);
    tool({"plutil", "-lint", versioned}, true);
    reserve(census, options.declaration, layouts, versioned, reservation);
    fs::create_directory(join(unbound, "Contents/MacOS"), error);
    link(options.arch, options.objects, join(unbound, "Contents/MacOS/" + options.name),
         reservation);
    // Staging refuses an existing output, so the last one goes first.
    fs::remove_all(options.output, error);
    stage(unbound, options.declaration, layouts, options.output);
}

// Check each staged kext in collection, printing a line for each that passes and
// a last one when all do.
void checkPrelinkedCommand(const std::string &path, const std::vector<std::string> &kexts) {
    Collection collection(readFile(path));
    for (const std::string &kext : kexts) {
        std::string plist = join(kext, "Contents/Info.plist");
        Plist info = parsePlist(readFile(plist), plist);
        const Plist *executable =
                info.kind == Plist::Kind::Dict ? info.find("CFBundleExecutable") : nullptr;
        const Plist *identifier =
                info.kind == Plist::Kind::Dict ? info.find("CFBundleIdentifier") : nullptr;
        if (!executable || executable->kind != Plist::Kind::String || !identifier ||
            identifier->kind != Plist::Kind::String)
            throw Error(plist + " names no CFBundleExecutable and CFBundleIdentifier");
        Image staged(readFile(join(kext, "Contents/MacOS/" + executable->text)));
        size_t count = checkPrelinked(collection, identifier->text, staged);
        std::cout << "prelink: " << identifier->text << ": manifest intact, " << count
                  << " sites at their placeholders\n";
    }
    std::string name = path.substr(path.rfind('/') + 1);
    std::cout << "prelink: OK: every listed consumer is bindable in " << name << "\n";
}

bool isOption(const std::string &arg) {
    return arg.size() > 1 && arg[0] == '-';
}

bool isHelp(const std::string &arg) {
    return arg == "-h" || arg == "--help";
}

// A command's operands, which must be exactly count, or at least count when more
// may follow.
std::vector<std::string> operands(const std::vector<std::string> &args, size_t count,
                                  bool more = false) {
    for (const std::string &arg : args) {
        if (isOption(arg))
            throw UsageError{"unrecognized option " + arg};
    }
    if (args.size() < count || (!more && args.size() > count))
        throw UsageError{"expected " + std::string(more ? "at least " : "") +
                         std::to_string(count) + " operands"};
    std::vector<std::string> paths;
    for (const std::string &arg : args)
        paths.push_back(normalized(arg));
    return paths;
}

BuildOptions buildOptions(const std::vector<std::string> &args) {
    static const char *const names[] = {"name",       "version", "arch",  "declaration",
                                        "info-plist", "work",    "output"};
    std::map<std::string, std::string> values;
    size_t at = 0;
    for (; at < args.size() && isOption(args[at]) && args[at] != "--"; ++at) {
        std::string option = args[at].substr(2), value;
        size_t equals = option.find('=');
        bool joined = equals != std::string::npos;
        if (joined) {
            value = option.substr(equals + 1);
            option = option.substr(0, equals);
        }
        if (args[at].compare(0, 2, "--") != 0 ||
            std::find(std::begin(names), std::end(names), option) == std::end(names))
            throw UsageError{"unrecognized option " + args[at]};
        if (!joined) {
            if (++at == args.size())
                throw UsageError{"option --" + option + " needs a value"};
            value = args[at];
        }
        values[option] = value;
    }
    std::string missing;
    for (const char *name : names) {
        if (!values.count(name))
            missing += (missing.empty() ? "--" : ", --") + std::string(name);
    }
    if (!missing.empty())
        throw UsageError{"the following options are required: " + missing};
    BuildOptions options;
    options.name = values["name"];
    options.version = values["version"];
    options.declaration = normalized(values["declaration"]);
    options.infoPlist = normalized(values["info-plist"]);
    options.work = normalized(values["work"]);
    options.output = normalized(values["output"]);
    // ARCH is one argument holding every target flag, split as a shell splits words.
    const std::string &arch = values["arch"];
    for (size_t first = arch.find_first_not_of(" \t\n\r\f\v"); first != std::string::npos;) {
        size_t last = arch.find_first_of(" \t\n\r\f\v", first);
        options.arch.push_back(arch.substr(first, last - first));
        first = last == std::string::npos ? last : arch.find_first_not_of(" \t\n\r\f\v", last);
    }
    // The objects, then any compiler flags after a --.
    for (; at < args.size() && args[at] != "--"; ++at)
        options.objects.push_back(args[at]);
    if (at < args.size())
        options.flags.assign(args.begin() + static_cast<ptrdiff_t>(at) + 1, args.end());
    if (options.objects.empty())
        throw UsageError{"build needs at least one object"};
    return options;
}

// Run the command args name, returning graft-kext's exit status.
int dispatch(const std::vector<std::string> &args) {
    if (args.empty())
        throw UsageError{"a command is required"};
    const std::string &command = args[0];
    std::vector<std::string> rest(args.begin() + 1, args.end());
    if (isHelp(command) || (!rest.empty() && isHelp(rest[0]))) {
        std::cout << Usage << Description;
        return 0;
    }
    if (command == "build") {
        build(buildOptions(rest));
    } else if (command == "probe") {
        // The compiler flags follow the two operands, after a -- that may lead them.
        if (rest.size() < 2)
            throw UsageError{"expected 2 operands"};
        std::vector<std::string> paths = operands({rest[0], rest[1]}, 2);
        size_t first = rest.size() > 2 && rest[2] == "--" ? 3 : 2;
        probe(paths[0], paths[1], {rest.begin() + static_cast<ptrdiff_t>(first), rest.end()});
    } else if (command == "reserve") {
        std::vector<std::string> paths = operands(rest, 5);
        reserve(paths[0], paths[1], paths[2], paths[3], paths[4]);
    } else if (command == "stage") {
        std::vector<std::string> paths = operands(rest, 4);
        stage(paths[0], paths[1], paths[2], paths[3]);
    } else if (command == "check-prelinked") {
        std::vector<std::string> paths = operands(rest, 2, true);
        checkPrelinkedCommand(paths[0], {paths.begin() + 1, paths.end()});
    } else {
        throw UsageError{"unknown command " + command};
    }
    return 0;
}

} // namespace
} // namespace graft::tool

int main(int argc, char **argv) {
    using namespace graft::tool;
    std::vector<std::string> args(argv + 1, argv + argc);
    std::string prefix = !args.empty() && args[0] == "check-prelinked" ? "prelink" : "consumer";
    try {
        return dispatch(args);
    } catch (const UsageError &error) {
        std::cout.flush();
        std::cerr << Usage << "graft-kext: error: " << error.reason << "\n";
        return 2;
    } catch (const Error &error) {
        std::cout.flush();
        std::cerr << prefix << ": " << error.what() << "\n";
        return 1;
    } catch (const std::filesystem::filesystem_error &error) {
        std::cout.flush();
        std::cerr << prefix << ": " << error.what() << "\n";
        return 1;
    }
}
