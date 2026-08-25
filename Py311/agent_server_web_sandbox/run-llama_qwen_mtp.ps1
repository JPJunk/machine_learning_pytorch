# powershell -ExecutionPolicy Bypass -File run-llama_qwen.ps1
# run-llama.ps1 -ExecutionPolicy Bypass -File run-llama.ps1

$mask = 0x0FFF   # CPUs 0-11 (P-cores only)

# Mallin polut muuttujissa selkeyden vuoksi
$modelPath = "C:\Models\LuffyTheFox\Qwen3.6-35B-A3B-Uncensored-Genesis-V2-APEX-MTP-GGUF\Qwen3.6-35B-A3B-Uncensored-Genesis-MTP-APEX-Compact.gguf"
$mmprojPath = "C:\Models\LuffyTheFox\Qwen3.6-35B-A3B-Uncensored-Genesis-V2-APEX-MTP-GGUF\mmproj-Qwen3.6-35B-A3B-Uncensored-Genesis-f16.gguf"

# Launch llama-server in a new CMD window -c 131072
cmd /c start "" ^
cmd /c ".\llama-server.exe -m `"$modelPath`" --mmproj `"$mmprojPath`" -t 6 -ngl 0 -c 65536 -b 512 --ubatch-size 256 --flash-attn on --mlock -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 5001 --jinja --temp 0.6 --top-k 20 --top-p 0.95 --presence-penalty 1.5 --repeat-penalty 1.0"

# Wait for llama-server to spawn
Start-Sleep -Seconds 2

# Find the process
$proc = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue

if ($proc) {
    $proc.ProcessorAffinity = $mask
    Write-Host "Affinity applied to CPUs 0-11 (P-cores only)" -ForegroundColor Green
} else {
    Write-Host "Could not find llama-server process" -ForegroundColor Red
}
