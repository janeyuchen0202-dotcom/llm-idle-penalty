# Records CPU (and GPU engine) load every 1-2 s while a test round runs. Killed by idle_test.py.
param([string]$Mode = "gpu", [string]$Out)
$ErrorActionPreference = "SilentlyContinue"
# Format numbers with the invariant culture, so a locale with a decimal comma cannot break the CSV columns (added 2026-10-16)
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
$interval = 1000
if ($Mode -eq "cpu") { $interval = 2000 }
"time,cpu_utility,cpu_performance,cpu_freq_mhz,gpu3d_max,gpu3d_sum" | Out-File -FilePath $Out -Encoding ascii
while ($true) {
    $t = (Get-Date).ToString("o")
    $p = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
    $mx = ""
    $sm = ""
    if ($Mode -ne "cpu") {
        $g = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object { $_.Name -like "*engtype_3D*" }
        $mx = 0
        $sm = 0
        foreach ($e in $g) {
            $v = [double]$e.UtilizationPercentage
            $sm += $v
            if ($v -gt $mx) { $mx = $v }
        }
    }
    "$t,$($p.PercentProcessorUtility),$($p.PercentProcessorPerformance),$($p.ProcessorFrequency),$mx,$sm" | Out-File -FilePath $Out -Append -Encoding ascii
    Start-Sleep -Milliseconds $interval
}
