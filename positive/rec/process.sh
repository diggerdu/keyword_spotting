sox -r 16k -e signed -b 16 test.raw test.wav
play -r 16k -e signed -b 16 dxj1.raw
./addnoise -i input.list -o output.list -n subway.raw -s 20