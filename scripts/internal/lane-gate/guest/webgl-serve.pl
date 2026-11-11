#!/usr/bin/perl
# webgl-serve.pl PORT DIR RESULT SECONDS: the lane gate's page server, in the guest. Serves
# DIR/webgl.html on 127.0.0.1:PORT, writes the query of the first GET /result?... it is sent,
# URL-decoded, to RESULT, and exits; or exits 1 after SECONDS with no result. Loopback only:
# the page loads nothing from anywhere else.
#
# Each connection is served by a child of its own. Safari 16.6 (macOS 13) opens connections it
# sends no request on; a server that read its connections in turn waited on such a one while the
# page's own requests queued behind it, and the result never came.
use strict;
use warnings;
use IO::Socket::INET;
use POSIX ':sys_wait_h';

my ($port, $dir, $result, $seconds) = @ARGV;
die "usage: webgl-serve.pl PORT DIR RESULT SECONDS\n" unless defined $seconds;
my $server = IO::Socket::INET->new(LocalAddr => '127.0.0.1', LocalPort => $port, Listen => 8, ReuseAddr => 1)
    or die "webgl-serve: cannot listen on $port: $!\n";
my $parent = $$;
my %children;
sub finish {
	kill 'TERM', keys %children;
	exit $_[0];
}
$SIG{CHLD} = sub { while ((my $k = waitpid(-1, WNOHANG)) > 0) { delete $children{$k} } };
$SIG{USR1} = sub { finish(0) };
$SIG{ALRM} = sub { print STDERR "webgl-serve: no result in $seconds s\n"; finish(1) };
alarm $seconds;
while (1) {
	my $c = $server->accept;
	unless ($c) {
		next if $!{EINTR};
		die "webgl-serve: accept: $!\n";
	}
	my $pid = fork;
	die "webgl-serve: fork: $!\n" unless defined $pid;
	if ($pid) {
		$children{$pid} = 1;
		close $c;
		next;
	}
	$SIG{$_} = 'DEFAULT' for qw(CHLD USR1 ALRM);
	close $server;
	alarm $seconds;
	serve($c);
	exit 0;
}

# One connection's request, in a child: the page, or its result, which the first child to bring
# one writes to RESULT (made whole under another name, then linked to RESULT), telling the parent
# to end.
sub serve {
	my ($c) = @_;
	my $line = <$c> // '';
	while (my $h = <$c>) { last if $h =~ /^\r?\n$/ }
	my ($path) = $line =~ m{^GET (\S+) HTTP};
	$path //= '';
	if ($path eq '/webgl.html' && open(my $f, '<', "$dir/webgl.html")) {
		local $/;
		my $body = <$f>;
		print $c "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nCache-Control: no-store\r\n"
		    . "Content-Length: " . length($body) . "\r\n\r\n$body";
	} elsif ($path =~ m{^/result\?(.*)$}) {
		(my $q = $1) =~ s/%([0-9A-Fa-f]{2})/chr(hex($1))/ge;
		print $c "HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok";
		close $c;
		my $part = "$result.$$";
		open(my $o, '>', $part) or die "webgl-serve: $part: $!\n";
		print $o "$q\n";
		close $o;
		link($part, $result);
		unlink $part;
		kill 'USR1', $parent;
		return;
	} else {
		print $c "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n";
	}
	close $c;
}
