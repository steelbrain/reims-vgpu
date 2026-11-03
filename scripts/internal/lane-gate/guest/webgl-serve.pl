#!/usr/bin/perl
# webgl-serve.pl PORT DIR RESULT SECONDS: the lane gate's page server, in the guest. Serves
# DIR/webgl.html on 127.0.0.1:PORT, writes the query of the first GET /result?... it is sent,
# URL-decoded, to RESULT, and exits; or exits 1 after SECONDS with no result. Loopback only:
# the page loads nothing from anywhere else.
use strict;
use warnings;
use IO::Socket::INET;

my ($port, $dir, $result, $seconds) = @ARGV;
die "usage: webgl-serve.pl PORT DIR RESULT SECONDS\n" unless defined $seconds;
my $server = IO::Socket::INET->new(LocalAddr => '127.0.0.1', LocalPort => $port, Listen => 8, ReuseAddr => 1)
    or die "webgl-serve: cannot listen on $port: $!\n";
$SIG{ALRM} = sub { print STDERR "webgl-serve: no result in $seconds s\n"; exit 1 };
alarm $seconds;
while (my $c = $server->accept) {
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
		open(my $o, '>', $result) or die "webgl-serve: $result: $!\n";
		print $o "$q\n";
		close $o;
		exit 0;
	} else {
		print $c "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\n\r\n";
	}
	close $c;
}
