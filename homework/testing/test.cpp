#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <omp.h>

using namespace cv;
using namespace std;

void Read_Raw(string file_path, unsigned char *output, int height, int width)
{
  char *temp = &file_path[0];
  FILE *inputFile;
  inputFile = fopen(temp, "rb");
  fread(output, 1, height * width, inputFile);
  fclose(inputFile);
}
//寫入raw檔之副程式以方便之後撰寫程式
void Write_Raw(string file_path, unsigned char *input, int height, int width)
{
  char *temp = &file_path[0];
  FILE *outputFile;
  outputFile = fopen(temp, "w");
  fwrite(input, 1, height * width, outputFile);
  fclose(outputFile);
}
float DFT_TRANS(Mat inputMat, Mat outputMat1, Mat outputMat2, string name)
{
  clock_t start_time, end_time;
  float total_time = 0;
  start_time = clock();
  printf("DFT_start\n");
  string outputfilepath = "../data/output/";
  int M = inputMat.rows;
  int N = inputMat.cols;
  outputMat2.create(M, N, CV_64FC1);
  outputMat1.create(M, N, CV_64FC1);
  Mat outputTemp1(M, N, CV_64FC1);
  Mat outputTemp2(M, N, CV_64FC1);
  Mat trans(M, N, CV_64FC1);
  Mat displayMat1(M, N, CV_8UC1);
  Mat displayMat2(M, N, CV_8UC1);
  Mat displayMat3(M, N, CV_8UC1);
  Mat displayMat4(M, N, CV_8UC1);
  double comp = 0;
  complex<double> z;
  complex<double> data;
  complex<double> CN = N;
  double dataR=0,dataI=0;
  complex<double> array[M][N] = {0};

  for (int x = 0; x < M; x++)
  {
    for (int y = 0; y < N; y++)
    {
      inputMat.at<double>(x, y) = inputMat.at<double>(x, y) * pow((-1), x + y);
    }
  }
   #pragma omp parallel for
  for (int x = 0; x < M; x++)
  {
    #pragma omp parallel for private(data)
    for (int v = 0; v < N; v++)
    {
      data = 0;
      //
      //#pragma omp parallel for reduction(+:dataR,dataI)
      for (float y = 0; y < N; y++)
      {
        complex<double> temp = inputMat.at<double>(x, y);
        complex<double>z = -1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[x][(int)(v)] = data / CN;
    }
  }
  //
   #pragma omp parallel for
  for (int v = 0; v < M; v++)
  {
    #pragma omp parallel for private(data)
    for (int u = 0; u < N; u++)
    {
      data = 0;
      //
      for (float x = 0; x < M; x++)
      {
        complex<double> temp = array[(int)(x)][v];
        complex<double>z = -1i * (2 * M_PI) * ((u * x) / M);
        data = data + exp(z) * temp;
      }
      outputMat1.at<double>(u, v) = data.real() / M;
      outputMat2.at<double>(u, v) = data.imag() / M;
    }
  }
  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      comp = pow(outputMat1.at<double>(u, v), 2) + pow(outputMat2.at<double>(u, v), 2);
      trans.at<double>(u, v) = abs(pow(comp, 0.5));
    }
  }
  outputTemp1 = abs(outputMat1);
  outputTemp2 = abs(outputMat2);
  //normalize(outputTemp1, outputTemp1, 0, 1, NORM_MINMAX);
  //normalize(outputTemp2, outputTemp2, 0, 1, NORM_MINMAX);
  //normalize(trans, trans, 0, 1, NORM_MINMAX);

  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      displayMat1.at<uchar>(u, v) = 255 * pow(outputTemp1.at<double>(u, v), 0.4);
      displayMat2.at<uchar>(u, v) = 255 * pow(outputTemp2.at<double>(u, v), 0.4);
      displayMat3.at<uchar>(u, v) = 255 * pow(trans.at<double>(u, v), 0.3);
    }
  }
  // DFT phase
  // for (int u = 0; u < M; u++)
  // {
  //   for (int v = 0; v < N; v++)
  //     trans.at<double>(u, v) = (atan2(outputMat2.at<double>(u, v), outputMat1.at<double>(u, v)) + M_PI) / (2 * M_PI);
  // }
  // trans.convertTo(displayMat4, CV_8UC1, 255, 0);
  end_time = clock();
  total_time = end_time - start_time;
  imwrite(outputfilepath + name + "_DFT_Real.png", displayMat1);
  imwrite(outputfilepath + name + "_DFT_Imag.png", displayMat2);
  imwrite(outputfilepath + name + "_DFT_Magt.png", displayMat3);
  //imwrite(outputfilepath + name + "_DFT_Phas.png", displayMat4);

  // imshow("Real", displayMat1);
  // imshow("Imag", displayMat2);
  // imshow("Mag", displayMat3);
  // //imshow("Phas32", trans);
  // waitKey(0);
  // destroyAllWindows();
  return total_time;
}
float IDFT_TRANS(Mat inputMat1, Mat inputMat2, Mat outputMat, String name)
{
  
  clock_t start_time, end_time;
  float total_time = 0;
  start_time = clock();
  printf("IDFT_start\n");
  string outputfilepath = "../data/output/";
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  outputMat.create(M, N, CV_64FC1);
  Mat trans(M, N, CV_64FC1);
  double comp = 0;
  //complex<double> z;
  complex<double> dataV[M]={0};
  complex<double> data=0;
  complex<double> array[M][N] = {0};
  //#pragma omp parallel for
  #pragma omp parallel for
  for (int u = 0; u < M; u++)
  {
    #pragma omp parallel for private(data)
    for (int y = 0; y < N; y++)
    {
      data = 0;
      for (float v = 0; v < N; v++)
      {
        complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
        complex<double>z = 1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[u][(int)(y)] = data; // * pow((-1), (x + y)
    }
  }
  #pragma omp parallel for
  for (int y = 0; y < M; y++)
  {
    #pragma omp parallel for private(data)
    for (int x = 0; x < M; x++)
    {
      data = 0;
      //
      
      for (float u = 0; u < N; u++)
      {
        complex<double> temp = array[(int)(u)][y];
        complex<double>z = 1i * (2 * M_PI) * ((u * x) / M);
        data = data + exp(z) * temp;
      }
      outputMat.at<double>(x, y) = data.real() * pow((-1), (x + y));
    }
  }
  //trans.convertTo(outputMat, CV_64FC1, 1, 0);
  //outputMat = trans;
  end_time = clock();
  total_time = end_time - start_time;
  // imshow("IDFT", outputMat);
  // waitKey(0);
  // destroyAllWindows();
  //imwrite(outputfilepath + name + "_IDFT_Real.png", outputMat);
  return total_time;
}
int main() {
  unsigned char rawfile[256 * 256];
  unsigned char rawfilebig[512 * 512];
  unsigned long list[256];
  unsigned int cdf1[256];
  string inputfilepath = "../data/input/";
  string outputfilepath = "../data/output/";
  string inputstring;
  double start_time, end_time;
  double total_time = 0;
  float tt=0;
  double sum ;
  //#pragma omp parallel for

  Mat house_org(512, 512, CV_8UC1);
  Mat test(512, 512, CV_64FC1);
  Mat DFT_Real(512, 512, CV_64FC1);
  Mat DFT_Imag(512, 512, CV_64FC1);
  Mat IDFT_Real(512, 512, CV_64FC1);
  Mat out(512,512,CV_8UC1);

  Read_Raw(inputfilepath + "house512.raw", rawfilebig, 512, 512);
  for (int i = 0; i < house_org.rows; i++)
  {
    for (int j = 0; j < house_org.cols; j++)
      house_org.at<uchar>(i, j) = rawfilebig[i * house_org.cols + j];
  }
  house_org.convertTo(test,CV_64FC1,1,0);

  start_time = omp_get_wtime();
  DFT_TRANS(test,DFT_Real,DFT_Imag,"test");
  IDFT_TRANS(DFT_Real,DFT_Imag,IDFT_Real,"test");
  end_time = omp_get_wtime();
  IDFT_Real.convertTo(out,CV_8UC1,1,0);

  imshow("test",out);
  waitKey(0);
  destroyAllWindows();




  // #pragma omp parallel for reduction(+:sum)
  // for(int j=0;j<10;j++)
  // {
  //   sum = sum + (double)(j);
  // }
    
  // printf("sum=%f\n",sum);



  total_time= end_time-start_time;
  printf("%f\n",total_time);
  total_time=0;



  return 0;




}
