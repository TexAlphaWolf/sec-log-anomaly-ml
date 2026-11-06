<#
.SYNOPSIS
  Exports Windows Security events to a JSON-lines file that loganomaly reads.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\export-windows-events.ps1 -Days 14
  loganomaly train --windows windows-security.jsonl --out model

.NOTES
  Reading the Security log needs an elevated PowerShell (Run as administrator).
  The file contains account names and addresses: keep it private, do not commit it.
#>
param(
    [int]$Days = 14,
    [string]$Out = "windows-security.jsonl",
    [string]$LogName = "Security",
    [int[]]$Ids = @(4624, 4625, 4672, 4698, 4720, 4728, 4732, 4756, 1102)
)

$ErrorActionPreference = "Stop"
$start = (Get-Date).AddDays(-$Days)
$count = 0
$writer = New-Object System.IO.StreamWriter((Join-Path (Get-Location) $Out), $false, (New-Object System.Text.UTF8Encoding($false)))
try {
    Get-WinEvent -FilterHashtable @{ LogName = $LogName; Id = $Ids; StartTime = $start } -ErrorAction SilentlyContinue |
        ForEach-Object {
            $xml = [xml]$_.ToXml()
            $data = [ordered]@{}
            foreach ($n in $xml.Event.EventData.Data) {
                if ($n.Name) { $data[$n.Name] = [string]$n.'#text' }
            }
            $line = [pscustomobject]@{
                TimeCreated = $_.TimeCreated.ToUniversalTime().ToString("o")
                EventID     = $_.Id
                Computer    = $_.MachineName
                Channel     = $_.LogName
                Data        = $data
            } | ConvertTo-Json -Compress -Depth 4
            $writer.WriteLine($line)
            $count++
        }
}
finally {
    $writer.Close()
}
Write-Host "Wrote $count events from the last $Days days to $Out"
if ($count -eq 0) {
    Write-Warning "No events found. Is PowerShell running as administrator?"
}
