# tflite surgeon
# command
- activate docker
    - `sudo docker run --rm -it -v `pwd`:/home/user/workdir ghcr.io/pinto0309/tflite2json2tflite:latest`
- convert tflite to json
    - `./flatc -t --strict-json --defaults-json -o ./workdir ./schema.fbs -- ./workdir/models/scrfd500m_256_320/scrfd500m_256_320_int8.tflite`
    - `./flatc -t --strict-json --defaults-json -o ./workdir ./schema.fbs -- ./workdir/models/scrfd500m_128_160/scrfd500m_128_160_int8.tflite`
- json modify
    - Const_104, Const_105, Const_106
        - scale: 0.047058823529411764
        - zero_point: 0
- convert json to tflite
    - `./flatc -o workdir/models/quantize_fix -b ./schema.fbs workdir/scrfd500m_256_320_int8.json`
    - `./flatc -o workdir/models/quantize_fix -b ./schema.fbs workdir/scrfd500m_128_160_int8.json`

# step by step
1. get tflite model.
2. reproduce the quantize problem
3. convert tflite to json
4. json modify
5. convert json to tflite
6. quantize problem check

# reference
- [tflite2json2tflite](https://github.com/PINTO0309/tflite2json2tflite)