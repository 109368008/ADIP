#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {



  // 讀取影像檔案
  Mat image;
  image = imread( "lena.png");

  // 檢查影像是否正確讀入
  /*if ( !image.data ) {
    printf("No image data n");
    return -1;
  }*/

  // 建立視窗
  namedWindow("Display Image", WINDOW_AUTOSIZE);

  // 用視窗顯示影像
  imshow("Display Image", image);

  // 顯示視窗，直到任何鍵盤輸入後才離開
  waitKey(0);

  return 0;
}