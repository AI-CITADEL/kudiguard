# Start the local N-ATLAS server (llama.cpp, CUDA) with the measured settings (docs/SPIKES.md S1, S3).
#   --cache-ram 2048 : host-RAM prompt cache; keeps shared prefixes cached across analyse/compose (saves ~150 ms/call)
#   -c 4096 : context budget (VRAM: 7.5 of 8 GB in use on an RTX 4070 Laptop)
#   -ngl 99 : all layers on the GPU
#   -np 1   : one slot; slot pinning saves <= ~35 ms/call (S3) and costs VRAM
#   --jinja : use the model's Llama-3 chat template
param(
    [string]$Model = "$PSScriptRoot\..\models\N-ATLaS-GGUF-Q4_K_M.gguf",
    [int]$Port = 8081
)
$server = Join-Path $PSScriptRoot "..\tools\llama\llama-server.exe"
& $server -m $Model -c 4096 -ngl 99 -np 1 --cache-ram 2048 --jinja --host 127.0.0.1 --port $Port
