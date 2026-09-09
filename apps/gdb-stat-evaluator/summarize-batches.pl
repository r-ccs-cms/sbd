#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Scalar::Util qw(looks_like_number);

my $output_path = '-';
my $help = 0;
GetOptions('output=s' => \$output_path, 'help|h' => \$help)
  or die "usage: summarize-batches.pl [--output FILE] CSV [CSV ...]\n";
if ($help) {
  print "usage: summarize-batches.pl [--output FILE] CSV [CSV ...]\n";
  exit 0;
}
@ARGV or die "at least one input CSV is required\n";

my @columns = qw(format_version calculation_id observable base_seed batch_id
  sample_count heatbath_cutoff reference_energy variance_estimate pt2_estimate
  mpi_size omp_threads elapsed_seconds);
my $expected_header = join(',', @columns);
my %groups;
my %seen_batch;

sub add_value {
  my ($stats, $value, $label) = @_;
  looks_like_number($value) or die "non-numeric $label: $value\n";
  ++$stats->{n};
  my $delta = $value - $stats->{mean};
  $stats->{mean} += $delta / $stats->{n};
  $stats->{m2} += $delta * ($value - $stats->{mean});
}

for my $path (@ARGV) {
  open my $input, '<', $path or die "cannot open $path: $!\n";
  my $header = <$input>;
  defined $header or die "empty CSV: $path\n";
  $header =~ s/\r?\n\z//;
  $header eq $expected_header or die "unexpected CSV header in $path\n";
  my $line_number = 1;
  while (my $line = <$input>) {
    ++$line_number;
    $line =~ s/\r?\n\z//;
    next if $line eq '';
    my @field = split /,/, $line, -1;
    @field == @columns
      or die "$path:$line_number: expected 13 columns\n";
    my %row;
    @row{@columns} = @field;
    $row{format_version} eq '1'
      or die "$path:$line_number: unsupported format version\n";
    $row{calculation_id} ne ''
      or die "$path:$line_number: empty calculation ID\n";
    my $duplicate = join("\0", $row{calculation_id}, $row{batch_id});
    !$seen_batch{$duplicate}++
      or die "$path:$line_number: duplicate calculation/batch ID\n";

    my $group = ($groups{$row{calculation_id}} //= {
      calculation_id => $row{calculation_id},
      observable => $row{observable},
      sample_count => $row{sample_count},
      heatbath_cutoff => $row{heatbath_cutoff},
      reference_energy => $row{reference_energy},
      variance => { n => 0, mean => 0, m2 => 0 },
      pt2 => { n => 0, mean => 0, m2 => 0 },
      elapsed => { n => 0, mean => 0, m2 => 0 },
      batch_count => 0,
    });
    for my $key (qw(observable sample_count heatbath_cutoff reference_energy)) {
      $row{$key} eq $group->{$key}
        or die "$path:$line_number: inconsistent $key for calculation $row{calculation_id}\n";
    }
    ++$group->{batch_count};
    add_value($group->{variance}, $row{variance_estimate}, 'variance')
      if $row{variance_estimate} ne '';
    add_value($group->{pt2}, $row{pt2_estimate}, 'PT2')
      if $row{pt2_estimate} ne '';
    add_value($group->{elapsed}, $row{elapsed_seconds}, 'elapsed time');
  }
  close $input or die "cannot close $path: $!\n";
}

my $output;
if ($output_path eq '-') {
  $output = *STDOUT;
} else {
  open $output, '>', $output_path
    or die "cannot open $output_path: $!\n";
}
print $output join(',', qw(format_version calculation_id observable batch_count
  sample_count heatbath_cutoff reference_energy variance_mean
  variance_sample_stddev variance_standard_error pt2_mean pt2_sample_stddev
  pt2_standard_error elapsed_seconds_total)), "\n";

sub summary_fields {
  my ($stats) = @_;
  return ('', '', '') if $stats->{n} == 0;
  my $stddev = $stats->{n} > 1
    ? sqrt($stats->{m2} / ($stats->{n} - 1)) : 0;
  my $error = $stddev / sqrt($stats->{n});
  return map { sprintf('%.17g', $_) } ($stats->{mean}, $stddev, $error);
}

for my $id (sort keys %groups) {
  my $group = $groups{$id};
  my @variance = summary_fields($group->{variance});
  my @pt2 = summary_fields($group->{pt2});
  my $elapsed_total = $group->{elapsed}{mean} * $group->{elapsed}{n};
  print $output join(',', 1, $id, $group->{observable}, $group->{batch_count},
    $group->{sample_count}, $group->{heatbath_cutoff},
    $group->{reference_energy}, @variance, @pt2,
    sprintf('%.17g', $elapsed_total)), "\n";
}
close $output if $output_path ne '-';
