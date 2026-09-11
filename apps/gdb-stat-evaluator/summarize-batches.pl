#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use Scalar::Util qw(looks_like_number);
use POSIX qw(isfinite);

my $output_path = '-';
my @logs;
my $profile_output;
my $rank_output;
my $help = 0;
my $usage = "usage: summarize-batches.pl [--output FILE] [CSV ...]\n"
  . "       [--log FILE ... --profile-output FILE [--rank-output FILE]]\n";
GetOptions('output=s' => \$output_path, 'help|h' => \$help,
           'log=s@' => \@logs, 'profile-output=s' => \$profile_output,
           'rank-output=s' => \$rank_output) or die $usage;
if ($help) { print $usage; exit 0; }
@ARGV || @logs or die $usage;
@logs && !defined($profile_output)
  and die "--log requires --profile-output\n";
!@logs && (defined($profile_output) || defined($rank_output))
  and die "diagnostic outputs require --log\n";
my @outputs = (@ARGV ? ($output_path) : (),
               defined($profile_output) ? ($profile_output) : (),
               defined($rank_output) ? ($rank_output) : ());
my %output_seen;
for my $path (@outputs) {
  !$output_seen{$path}++ or die "output paths must be distinct\n";
  !grep { $_ eq $path } (@ARGV, @logs)
    or die "output must not overwrite an input: $path\n";
}

my @columns = qw(format_version calculation_id observable base_seed batch_id
  sample_count heatbath_cutoff reference_energy variance_estimate pt2_estimate
  mpi_size omp_threads elapsed_seconds);
my $expected_header = join(',', @columns);
my %groups;
my %seen_batch;
my %csv_batches;

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
    $csv_batches{$duplicate} = { %row };
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

my $diagnostics = read_diagnostics(\@logs, \%csv_batches, scalar(@ARGV));
if (@ARGV) {
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

}
write_diagnostics($diagnostics, $profile_output, $rank_output) if @logs;

# Only rank rows are authoritative: profile_summary/timing_summary are ignored.
sub read_diagnostics {
  my ($paths, $csv, $check_csv) = @_;
  my @profile = qw(parents membership lookup_entries lookup_bytes sampled_parents
                   draws generated_records received_records unique_children external_children);
  my @timing = qw(distribution sampling expansion hash_observable hash hash_pack
                  hash_mpi hash_receive_sort child_sort local_observable observable_allreduce total);
  my %metrics = (profile => \@profile, timing => \@timing);
  my %batches;
  for my $path (@$paths) {
    open my $input, '<', $path or die "cannot open $path: $!\n";
    my ($id, $size);
    my $line_number = 0;
    while (my $line = <$input>) {
      ++$line_number;
      $line =~ s/\r?\n\z//;
      if ($line =~ /^# FCIDUMP:/) { undef $id; undef $size; }
      if ($line =~ /^# calculation ID: (.+)$/) { $id = $1; undef $size; }
      if ($line =~ /^# MPI size: (\d+)$/) { $size = $1; }
      next unless $line =~ /sbd::stats: (profile|timing)\s+(.*)$/;
      my ($kind, $rest) = ($1, $2);
      my $where = "$path:$line_number";
      defined($id) && defined($size) && $size > 0
        or die "$where: missing calculation ID or MPI size metadata\n";
      my %field;
      pos($rest) = 0;
      while ($rest =~ /\G\s*(\w+)=("(?:[^"\\]|\\.)*"|[^\s]+)(?:\s+|$)/gc) {
        my ($key, $value) = ($1, $2);
        !exists($field{$key}) or die "$where: repeated field $key\n";
        if ($value =~ /^"(.*)"$/s) {
          $value = $1;
          $value =~ s/\\(.)/$1/gs;
        }
        $field{$key} = $value;
      }
      (pos($rest) // 0) == length($rest)
        or die "$where: malformed diagnostic fields\n";
      for my $key (qw(batch_id rank)) {
        defined($field{$key}) && $field{$key} =~ /^\d+$/
          or die "$where: missing or invalid $key (rank rows required)\n";
      }
      $field{rank} < $size or die "$where: rank outside MPI size\n";
      my $key = join("\0", $id, $field{batch_id});
      my $batch = ($batches{$key} //= {
        id => $id, batch => $field{batch_id}, size => $size, rows => {},
      });
      $batch->{size} == $size or die "$where: inconsistent MPI size\n";
      !exists($batch->{rows}{$kind}{$field{rank}})
        or die "$where: duplicate calculation/batch/kind/rank\n";
      if ($kind eq 'profile') {
        defined($field{host}) && length($field{host})
          or die "$where: missing host\n";
      }
      for my $metric (@{$metrics{$kind}}) {
        my $value = $field{$metric};
        defined($value) && looks_like_number($value) && isfinite(0 + $value) && $value >= 0
          or die "$where: missing or invalid $metric\n";
        $kind ne 'profile' || $value =~ /^\d+$/
          or die "$where: non-integer count $metric\n";
      }
      $batch->{rows}{$kind}{$field{rank}} = \%field;
    }
    close $input or die "cannot close $path: $!\n";
  }
  @$paths && !keys(%batches) and die "no rank profile/timing rows found\n";
  for my $key (keys %batches) {
    my $batch = $batches{$key};
    for my $kind (qw(profile timing)) {
      for my $rank (0 .. $batch->{size} - 1) {
        exists($batch->{rows}{$kind}{$rank})
          or die "incomplete log: calculation $batch->{id}, batch $batch->{batch}, $kind rank $rank\n";
      }
    }
    my ($draws, $generated, $received) = (0, 0, 0);
    for my $rank (0 .. $batch->{size} - 1) {
      my $row = $batch->{rows}{profile}{$rank};
      $row->{sampled_parents} <= $row->{parents} &&
      $row->{sampled_parents} <= $row->{draws} &&
      $row->{external_children} <= $row->{unique_children} &&
      $row->{unique_children} <= $row->{received_records}
        or die "inconsistent profile counts in calculation $batch->{id}, batch $batch->{batch}\n";
      $draws += $row->{draws};
      $generated += $row->{generated_records};
      $received += $row->{received_records};
    }
    $generated == $received or die "generated/received totals differ\n";
    if ($check_csv) {
      my $row = $csv->{$key} or die "log batch has no matching CSV row\n";
      $row->{mpi_size} == $batch->{size} && $row->{sample_count} == $draws
        or die "log MPI size or draw count disagrees with CSV\n";
    }
  }
  return { batches => \%batches, metrics => \%metrics };
}

sub csv_field {
  my ($value) = @_;
  if ($value =~ /[",\r\n]/) { $value =~ s/"/""/g; return '"' . $value . '"'; }
  return $value;
}
sub open_output {
  my ($path) = @_;
  return *STDOUT if $path eq '-';
  open my $fh, '>', $path or die "cannot open $path: $!\n";
  return $fh;
}
sub write_diagnostics {
  my ($data, $summary_path, $rank_path) = @_;
  my $summary = open_output($summary_path);
  my $ranks = defined($rank_path) ? open_output($rank_path) : undef;
  print $summary "calculation_id,batch_id,kind,metric,rank_count,min,mean,max,max_over_mean\n";
  print $ranks "calculation_id,batch_id,rank,host,kind,metric,value\n" if defined($ranks);
  my $batches = $data->{batches};
  for my $key (sort { $batches->{$a}{id} cmp $batches->{$b}{id} ||
                     $batches->{$a}{batch} <=> $batches->{$b}{batch} } keys %$batches) {
    my $batch = $batches->{$key};
    for my $kind (qw(profile timing)) {
      for my $metric (@{$data->{metrics}{$kind}}) {
        my $stats = { n => 0, mean => 0, m2 => 0 };
        my ($min, $max);
        for my $rank (0 .. $batch->{size} - 1) {
          my $value = $batch->{rows}{$kind}{$rank}{$metric};
          add_value($stats, $value, $metric);
          $min = $value if !defined($min) || $value < $min;
          $max = $value if !defined($max) || $value > $max;
          if (defined($ranks)) {
            my $host = $batch->{rows}{profile}{$rank}{host};
            print $ranks join(',', map { csv_field($_) }
              ($batch->{id}, $batch->{batch}, $rank, $host, $kind, $metric, $value)), "\n";
          }
        }
        my $mean = $stats->{mean};
        my $ratio = $mean == 0 ? '' : sprintf('%.17g', $max / $mean);
        print $summary join(',', map { csv_field($_) }
          ($batch->{id}, $batch->{batch}, $kind, $metric, $batch->{size},
           $min, sprintf('%.17g', $mean), $max, $ratio)), "\n";
      }
    }
  }
  close $summary or die "cannot close $summary_path: $!\n" if $summary_path ne '-';
  if (defined($ranks) && $rank_path ne '-') {
    close $ranks or die "cannot close $rank_path: $!\n";
  }
}
