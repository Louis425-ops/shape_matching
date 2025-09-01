```bash
g++ -std=c++14 -O3 -fopenmp -Wall -Wno-sign-compare
  -march=native -I MIPP/ -I /usr/include/opencv4
  nut_detection.cpp line2Dup.cpp -lopencv_core
  -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -o
  nut_detector_new
```
