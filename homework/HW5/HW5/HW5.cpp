#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//讀取raw檔之副程式以方便之後撰寫程式
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
float MSE(Mat inputMat, Mat outputMat)
{
  double sum = 0;
  int count = 0, a;
  float mse = 0;
  for (int i = 0; i < inputMat.cols; i++)
  {
    for (int j = 0; j < inputMat.rows; j++)
    {
      a = outputMat.at<uchar>(j, i) - inputMat.at<uchar>(j, i);
      count++;
      sum = sum + (a * a);
    }
  }
  mse = (float)sum / count;
  return mse;
}
float PSNR(Mat inputMat, Mat outputMat)
{
  double sum = 0;
  int count = 0, a;
  float mse = 0, psnr = 0;
  for (int i = 0; i < inputMat.cols; i++)
  {
    for (int j = 0; j < inputMat.rows; j++)
    {
      a = outputMat.at<uchar>(j, i) - inputMat.at<uchar>(j, i);
      count++;
      sum = sum + (a * a);
    }
  }
  mse = (float)sum / count;

  psnr = 10 * log10((255 * 255) / mse);
  return psnr;
}
float DFT_TRANS(Mat inputMat, Mat outputMat1, Mat outputMat2, String name)
{
  clock_t start_time, end_time;
  float total_time = 0;
  start_time = clock();
  printf("DFT_start");
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

  complex<double> array[M][N] = {0};

  for (int x = 0; x < M; x++)
  {
    for (int y = 0; y < N; y++)
    {
      inputMat.at<double>(x, y) = inputMat.at<double>(x, y) * pow((-1), x + y);
    }
  }
  
  for (float x = 0; x < M; x++)
  {
    for (float v = 0; v < N; v++)
    {
      data = 0;
      for (float y = 0; y < N; y++)
      {
        complex<double> temp = inputMat.at<double>(x, y);
        z = -1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[(int)(x)][(int)(v)] = data/CN;
    }
  }

  for (float u = 0; u < M; u++)
  {
    for (float v = 0; v < N; v++)
    {
      data = 0;
      for (float x = 0; x < M; x++)
      {
        complex<double> temp = array[(int)(x)][(int)(v)];
        z = -1i * (2 * M_PI) * ((u * x) / M);
        data = data + exp(z) * temp;
      }
      outputMat1.at<double>(u, v) = data.real()/M;
      outputMat2.at<double>(u, v) = data.imag()/M;
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
  normalize(outputTemp1, outputTemp1, 0, 1, NORM_MINMAX);
  normalize(outputTemp2, outputTemp2, 0, 1, NORM_MINMAX);
  normalize(trans, trans, 0, 1, NORM_MINMAX);

  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      displayMat1.at<uchar>(u, v) = 255 * pow(outputTemp1.at<double>(u, v), 0.4);
      displayMat2.at<uchar>(u, v) = 255 * pow(outputTemp2.at<double>(u, v), 0.4);
      displayMat3.at<uchar>(u, v) = 255 * pow(trans.at<double>(u, v), 0.3);
    }
  }
  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
      trans.at<double>(u, v) = (atan2(outputMat2.at<double>(u, v), outputMat1.at<double>(u, v)) + M_PI) / (2 * M_PI);
  }
  trans.convertTo(displayMat4, CV_8UC1, 255, 0);
  end_time = clock();
  total_time = end_time - start_time;
  imwrite(outputfilepath + name + "_DFT_Real.png", displayMat1);
  imwrite(outputfilepath + name + "_DFT_Imag.png", displayMat2);
  imwrite(outputfilepath + name + "_DFT_Magt.png", displayMat3);
  imwrite(outputfilepath + name + "_DFT_Phas.png", displayMat4);
  imshow("Real", displayMat1);
  imshow("Imag", displayMat2);
  imshow("Mag", displayMat3);
  imshow("Phas32", trans);
  waitKey(0);
  destroyAllWindows();
  return total_time;
}
float IDFT_TRANS(Mat inputMat1, Mat inputMat2, Mat outputMat, String name)
{
  clock_t start_time, end_time;
  float total_time = 0;
  start_time = clock();
  printf("IDFT_start");
  string outputfilepath = "../data/output/";
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  outputMat.create(M, N, CV_8UC1);
  Mat trans(M, N, CV_64FC1);
  double comp = 0;
  complex<double> z;
  complex<double> data;
  complex<double> array[M][N] = {0};

  for (float u = 0; u < M; u++)
  {
    for (float y = 0; y < N; y++)
    {
      data = 0;
      for (float v = 0; v < N; v++)
      {
        complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
        z = 1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[(int)(u)][(int)(y)] = data;// * pow((-1), (x + y))
    }
    
  }
  for (float y = 0; y < M; y++)
  {
      for (float x = 0; x < M; x++)
      {
        data = 0;
        for (float u = 0; u < N; u++)
        {
          complex<double> temp = array[(int)(u)][(int)(y)];
          z = 1i * (2 * M_PI) * ((u * x) / M) ;
          data = data + exp(z) * temp;
        }
        trans.at<double>(x, y) = data.real() * pow((-1), (x + y));
      }
  }
  trans.convertTo(outputMat, CV_8UC1, 255, 0);
  end_time = clock();
  total_time = end_time - start_time;
  imshow("IDFT", outputMat);
  waitKey(0);
  destroyAllWindows();
  imwrite(outputfilepath + name + "_IDFT_Real.png", outputMat);
  return total_time;
}
float Opencv_DFT(Mat inputMat, Mat outputMat1, Mat outputMat2, String name)
{
  string outputfilepath = "../data/output/";
  clock_t start_time, end_time;
  float total_time = 0;
  start_time = clock();
  int M = getOptimalDFTSize(inputMat.rows); // 获得最佳DFT尺寸，为2的次方
  int N = getOptimalDFTSize(inputMat.cols);
  outputMat1.create(M, N, CV_8UC1);
  outputMat2.create(M, N, CV_8UC1);
  //printf("M=%d,N=%d\n", M, N);
  Mat padded;
  copyMakeBorder(inputMat, padded, 0, M - inputMat.rows, 0, N - inputMat.cols, BORDER_CONSTANT, Scalar::all(0));
  // opencv中的边界扩展函数，提供多种方式扩展
  Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)}; // Mat 数组，第一个为扩展后的图像，一个为空图像，
  Mat complexImg;

  merge(planes, 2, complexImg); // 合并成一个Mat

  dft(complexImg, complexImg); // FFT变换， dft需要一个2通道的Mat
  // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
  split(complexImg, planes); //分离通道， planes[0] 为实数部分，planes[1]为虚数部分

  magnitude(planes[0], planes[1], planes[0]); // 求模

  Mat mag = planes[0];
  mag += Scalar::all(1);
  log(mag, mag); // 模的对数

  // crop the spectrum, if it has an odd number of rows or columns
  mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2)); //保证偶数的边长

  int cx = mag.cols / 2;
  int cy = mag.rows / 2;
  //printf("cx=%d,cy=%d\n", cx, cy);
  // rearrange the quadrants of Fourier image
  //对傅立叶变换的图像进行重排，4个区块，从左到右，从上到下 分别为q0, q1, q2, q3
  // so that the origin is at the image center
  //  对调q0和q3, q1和q2
  Mat tmp;
  Mat q0(mag, Rect(0, 0, cx, cy));
  Mat q1(mag, Rect(cx, 0, cx, cy));
  Mat q2(mag, Rect(0, cy, cx, cy));
  Mat q3(mag, Rect(cx, cy, cx, cy));

  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  normalize(mag, mag, 0, 1, NORM_MINMAX); // 规范化值到 0~1 显示图片的需要
  end_time = clock();
  total_time = end_time - start_time;
  mag.convertTo(outputMat1, CV_8UC1, 255, 0);
  imwrite(outputfilepath + name + "_DFT.png", outputMat1);
  imshow("spectrum magnitude", mag);
  waitKey();
  destroyAllWindows();

  Mat ifft;
  idft(complexImg, ifft, DFT_REAL_OUTPUT);
  normalize(ifft, ifft, 0, 1, NORM_MINMAX);
  ifft.convertTo(outputMat2, CV_8UC1, 255, 0);
  imwrite(outputfilepath + name + "_IDFT.png", outputMat2);
  imshow("idft", ifft);
  waitKey();
  destroyAllWindows();

  return total_time;
}
//主程式開始
int main()
{
  //-------變數宣告區------------------------------------------------
  unsigned char rawfile[256 * 256];
  unsigned char rawfilebig[512 * 512];
  unsigned long list[256];
  unsigned int cdf1[256];
  float DFT_time = 0, IDFT_time = 0;
  float DFT_mse = 0, DFT_psnr = 0;
  string inputfilepath = "../data/input/";
  string outputfilepath = "../data/output/";
  string inputstring;
  //-------Mat宣告區------------------------------------------------
  Mat square_org(256, 256, CV_8UC1);
  Mat circle_org(256, 256, CV_8UC1);
  Mat rota_org(256, 256, CV_8UC1);
  Mat rect_org(256, 256, CV_8UC1);
  Mat DFT_Real(256, 256, CV_64FC1);
  Mat DFT_Imag(256, 256, CV_64FC1);
  Mat IDFT_Real(256, 256, CV_8UC1);
  Mat PIC_float(256, 256, CV_64FC1);

  //-------讀取檔案區------------------------------------------------
  Read_Raw(inputfilepath + "Square256.raw", rawfile, 256, 256);
  for (int i = 0; i < square_org.rows; i++)
  {
    for (int j = 0; j < square_org.cols; j++)
      square_org.at<uchar>(i, j) = rawfile[i * square_org.cols + j];
  }
  Read_Raw(inputfilepath + "rect256.raw", rawfile, 256, 256);
  for (int i = 0; i < rect_org.rows; i++)
  {
    for (int j = 0; j < rect_org.cols; j++)
    {
      rect_org.at<uchar>(i, j) = rawfile[i * rect_org.cols + j];
    }
  }
  Read_Raw(inputfilepath + "circle256.raw", rawfile, 256, 256);
  for (int i = 0; i < circle_org.rows; i++)
  {
    for (int j = 0; j < circle_org.cols; j++)
    {
      circle_org.at<uchar>(i, j) = rawfile[i * circle_org.cols + j];
    }
  }
  Read_Raw(inputfilepath + "square256_rota.raw", rawfile, 256, 256);
  for (int i = 0; i < rota_org.rows; i++)
  {
    for (int j = 0; j < rota_org.cols; j++)
    {
      rota_org.at<uchar>(i, j) = rawfile[i * rota_org.cols + j];
    }
  }

  while (inputstring != "quit")
  {
    printf("please enter the question number \n enter quit to exit\n");
    printf("menu \n rect square circle rota \n");
    printf(" cvrect cvsquare cvcircle cvrota\n");
    cin >> inputstring;
    if (inputstring == "test")
    {
      rect_org.convertTo(PIC_float, CV_64FC1, 1, 0);
      normalize(PIC_float, PIC_float, 0, 1, NORM_MINMAX);
      DFT_time = DFT_TRANS(PIC_float, DFT_Real, DFT_Imag, inputstring);
      printf("%f\n", DFT_time);
    }
    else if (inputstring == "rect")
    {
      rect_org.convertTo(PIC_float, CV_64FC1, 1, 0);
      normalize(PIC_float, PIC_float, 0, 1, NORM_MINMAX);
      DFT_time = DFT_TRANS(PIC_float, DFT_Real, DFT_Imag, inputstring);
      IDFT_time = IDFT_TRANS(DFT_Real, DFT_Imag, IDFT_Real, inputstring);
      DFT_mse = MSE(rect_org, IDFT_Real);
      DFT_psnr = PSNR(rect_org, IDFT_Real);
      DFT_time = DFT_time / 1000;
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << "ms " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "square")
    {
      square_org.convertTo(PIC_float, CV_64FC1, 1, 0);
      normalize(PIC_float, PIC_float, 0, 1, NORM_MINMAX);
      DFT_time = DFT_TRANS(PIC_float, DFT_Real, DFT_Imag, inputstring);
      IDFT_time = IDFT_TRANS(DFT_Real, DFT_Imag, IDFT_Real, inputstring);
      DFT_mse = MSE(square_org, IDFT_Real);
      DFT_psnr = PSNR(square_org, IDFT_Real);
      DFT_time = DFT_time / 1000;
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << "ms " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "circle")
    {
      circle_org.convertTo(PIC_float, CV_64FC1, 1, 0);
      normalize(PIC_float, PIC_float, 0, 1, NORM_MINMAX);
      DFT_time = DFT_TRANS(PIC_float, DFT_Real, DFT_Imag, inputstring);
      IDFT_time = IDFT_TRANS(DFT_Real, DFT_Imag, IDFT_Real, inputstring);
      DFT_mse = MSE(circle_org, IDFT_Real);
      DFT_psnr = PSNR(circle_org, IDFT_Real);
      DFT_time = DFT_time / 1000;
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << "ms " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "rota")
    {
      rota_org.convertTo(PIC_float, CV_64FC1, 1, 0);
      normalize(PIC_float, PIC_float, 0, 1, NORM_MINMAX);
      DFT_time = DFT_TRANS(PIC_float, DFT_Real, DFT_Imag, inputstring);
      IDFT_time = IDFT_TRANS(DFT_Real, DFT_Imag, IDFT_Real, inputstring);
      DFT_mse = MSE(rota_org, IDFT_Real);
      DFT_psnr = PSNR(rota_org, IDFT_Real);
      DFT_time = DFT_time / 1000;
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << "ms " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "cvrect")
    {
      DFT_time = Opencv_DFT(rect_org, DFT_Real, IDFT_Real, inputstring);
      DFT_mse = MSE(rect_org, IDFT_Real);
      DFT_psnr = PSNR(rect_org, IDFT_Real);
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << " us " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "cvcircle")
    {
      DFT_time = Opencv_DFT(circle_org, DFT_Real, IDFT_Real, inputstring);
      DFT_mse = MSE(circle_org, IDFT_Real);
      DFT_psnr = PSNR(circle_org, IDFT_Real);
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << " us " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "cvsquare")
    {
      DFT_time = Opencv_DFT(square_org, DFT_Real, IDFT_Real, inputstring);
      DFT_mse = MSE(square_org, IDFT_Real);
      DFT_psnr = PSNR(square_org, IDFT_Real);
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << " us " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
    else if (inputstring == "cvrota")
    {
      DFT_time = Opencv_DFT(rota_org, DFT_Real, IDFT_Real, inputstring);
      DFT_mse = MSE(rota_org, IDFT_Real);
      DFT_psnr = PSNR(rota_org, IDFT_Real);
      std::cout << "Execution time of the DFT of " << inputstring << " is " << DFT_time << " us " << std::endl;
      printf("MSE of input and output is %f\n", DFT_mse);
      printf("PSNR of input and output is %f\n", DFT_psnr);
    }
  }
  return 0;
}
