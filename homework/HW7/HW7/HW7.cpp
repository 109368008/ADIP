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

double ln(double x)
{
  double data = 0;
  data = log(x) / log(exp(1));
  return data;
}
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
double MSE(Mat inputMat, Mat outputMat)
{
  int M = inputMat.rows;
  int N = inputMat.cols;
  double sum = 0;
  int a;
  double mse = 0;
  for (int i = 0; i < M; i++)
  {

    for (int j = 0; j < N; j++)
    {
      a = outputMat.at<uchar>(j, i) - inputMat.at<uchar>(j, i);
      sum = sum + (a * a);
    }
    mse = mse + (sum / N);
    sum = 0;
  }
  mse = mse / M;
  return mse;
}
double PSNR(Mat inputMat, Mat outputMat)
{
  double sum = 0;
  int count = 0, a;
  double mse = 0, psnr = 0;
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
#pragma omp parallel for private(data)
    for (int v = 0; v < N; v++)
    {
      data = 0;
      for (float y = 0; y < N; y++)
      {
        complex<double> temp = inputMat.at<double>(x, y);
        complex<double> z = -1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[(int)(x)][(int)(v)] = data / CN;
    }
  }
  for (float v = 0; v < N; v++)
  {
#pragma omp parallel for private(data)
    for (int u = 0; u < M; u++)
    {
      data = 0;
      //
      for (float x = 0; x < M; x++)
      {
        complex<double> temp = array[(int)(x)][(int)(v)];
        complex<double> z = -1i * (2 * M_PI) * ((u * x) / M);
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
  complex<double> data = 0;
  complex<double> array[M][N] = {0};
  for (int u = 0; u < M; u++)
  {
#pragma omp parallel for private(data)
    for (int y = 0; y < N; y++)
    {
      data = 0;
      for (float v = 0; v < N; v++)
      {
        complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
        complex<double> z = 1i * (2 * M_PI) * ((v * y) / N);
        data = data + exp(z) * temp;
      }
      array[u][(int)(y)] = data; // * pow((-1), (x + y)
    }
  }
  for (int y = 0; y < M; y++)
  {
#pragma omp parallel for private(data)
    for (int x = 0; x < M; x++)
    {
      data = 0;
      for (float u = 0; u < N; u++)
      {
        complex<double> temp = array[(int)(u)][y];
        complex<double> z = 1i * (2 * M_PI) * ((u * x) / M);
        data = data + exp(z) * temp;
      }
      outputMat.at<double>(x, y) = data.real() * pow((-1), (x + y));
    }
  }
  //trans.convertTo(outputMat, CV_64FC1, 1, 0);
  //outputMat = trans;
  end_time = clock();
  total_time = end_time - start_time;
  //imshow("IDFT", outputMat);
  //waitKey(0);
  //destroyAllWindows();
  //imwrite(outputfilepath + name + "_IDFT_Real.png", outputMat);
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
void powerLaw(Mat inputMat, float gamma)
{
  int M = inputMat.rows;
  int N = inputMat.cols;
  for (int x = 0; x < M; x++)
  {
    for (int y = 0; y < N; y++)
    {
      inputMat.at<double>(x, y) = pow(inputMat.at<double>(x, y), gamma);
    }
  }
}
void GaussianFilter(Mat inputMat1, Mat inputMat2, int d0, int mode) //0=LPF 1=HPF
{
  string outputfilepath = "../data/output/";
  string name;
  string F;
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat display1(M, N, CV_8UC1);
  complex<double> G = 0;
  double min, max;
  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double dis = pow(pow(u - (M / 2.0), 2) + pow(v - (N / 2.0), 2), 0.5); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      if (mode == 0)                                                        //low pass
      {
        F = "LPF";
        G = exp(-(pow(dis, 2) / (pow(d0, 2) * 2)));
      }
      else if (mode == 1) //high pass
      {

        F = "HPF";
        G = 1 - (exp((pow(dis, 2) / (pow(d0, 2) * (-2)))) * 0.85);
        // if(u==256)
        // {
        //   printf("g = %f , d0=%d\n",G.real(),d0);
        // }
      }
      temp = temp * G;
      display1.at<uchar>(u, v) = G.real() * 255;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }

  // minMaxLoc(display1,&min,&max,NULL,NULL);
  // cout << "min" << min << "max" << max << endl;
  powerLaw(outputMat1, 0.3);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + "Gaussian" + F + "_d0=" + to_string(d0) + "_DFT_filter.png";
  imwrite(name, display1);
  name = outputfilepath + "Gaussian" + F + "_d0=" + to_string(d0) + "_DFT_Magt.png";
  imwrite(name, outputMat1);
}
void Idealfilter(Mat inputMat1, Mat inputMat2, int d0, int mode) //0=LPF 1=HPF
{
  string outputfilepath = "../data/output/";
  string name;
  string F;
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat display1(M, N, CV_8UC1);
  complex<double> G = 0;

  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double dis = sqrt(pow(u - (M / 2.0), 2) + pow(v - (N / 2.0), 2)); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      if (mode == 0)                                                    //low pass
      {
        if (dis > d0)
          G = 0;
        else
          G = 1;
        F = "LPF";
      }
      if (mode == 1) //high pass
      {
        if (dis > d0)
          G = 1;
        else
          G = 0;
        F = "HPF";
      }
      temp = temp * G;
      display1.at<uchar>(u, v) = G.real() * 255;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }
  powerLaw(outputMat1, 0.3);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + "ideal" + F + "_d0=" + to_string(d0) + "_DFT_filter.png";
  imwrite(name, display1);
  name = outputfilepath + "ideal" + F + "_d0=" + to_string(d0) + "_DFT_Magt.png";
  imwrite(name, outputMat1);
}
string Homofilter(Mat inputMat1, Mat inputMat2, double rL, double rH, double c, int d0)
{
  string outputfilepath = "../data/output/";
  string name;
  string para = to_string(((int)(rL * 10) / 10)) + "." + to_string(((int)(rL * 10) % 10)) + "_" + to_string((int)(rH * 10) / 10) + "." + to_string((int)(rH * 10) % 10) + "_";
  para = para + to_string((int)(c * 10) / 10) + "." + to_string((int)(c * 10) % 10) + "_" + to_string(d0);
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat outputMat2(M, N, CV_64FC1);
  Mat display1(M, N, CV_64FC1);
  double G = 0;
  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double dis = sqrt(pow(u - (M / 2.0), 2) + pow(v - (N / 2.0), 2)); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      G = ((rH - rL) * (1 - (exp(-c * (pow(dis, 2) / pow(d0, 2))))) + rL);
      temp = temp * G;
      display1.at<double>(u, v) = G / rH;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }
  powerLaw(outputMat1, 0.3);
  display1.convertTo(display1, CV_8UC1, 255, 0);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + "Homofilter" + para + "_Filter.png";
  imwrite(name, display1);
  name = outputfilepath + "Homofilter" + para + "_Magt.png";
  imwrite(name, outputMat1);
  return para;
}
void BandRejectFilter(Mat inputMat1, Mat inputMat2, int d0, int w, string inputstr)
{
  string outputfilepath = "../data/output/";
  string name;
  string F;
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat display1(M, N, CV_8UC1);
  complex<double> B = 0;

  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double dis = sqrt(pow(u - (M / 2.0), 2) + pow(v - (N / 2.0), 2)); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      if (dis < d0 - (w / 2.0))
      {
        B = 1;
      }
      else if ((d0 - (w / 2.0)) <= dis && dis <= (d0 + (w / 2.0)))
      {
        B = 0;
        printf("%f\n", B.real());
      }
      else if (dis > d0 + (w / 2.0))
      {
        B = 1;
      }
      temp = temp * B;
      display1.at<uchar>(u, v) = B.real() * 255;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }
  powerLaw(outputMat1, 0.3);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + inputstr + "ideal" + F + "_d0=" + to_string(d0) + "_DFT_filter.png";
  imwrite(name, display1);
  name = outputfilepath + inputstr + "ideal" + F + "_d0=" + to_string(d0) + "_DFT_Magt.png";
  imwrite(name, outputMat1);
}
void NotchFilter(Mat inputMat1, Mat inputMat2, int d0, int u0, int v0, string inputname, string inputstr)
{
  string outputfilepath = "../data/output/";
  string name;
  string F;
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat display1(M, N, CV_8UC1);
  complex<double> G = 0;

  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double duv1 = sqrt(pow(u - (M / 2.0) - u0, 2) + pow(v - (N / 2.0) - v0, 2)); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      double duv2 = sqrt(pow(u - (M / 2.0) + u0, 2) + pow(v - (N / 2.0) + v0, 2));
      G = 1 - (exp(-0.5 * ((duv1 * duv2) / pow(d0, 2))) * 0.95);
      temp = temp * G;
      display1.at<uchar>(u, v) = G.real() * 255;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }
  powerLaw(outputMat1, 0.3);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + inputstr + "gaussian" + F + "_d0=" + to_string(d0) + "_DFT_" + inputname + ".png";
  imwrite(name, display1);
  name = outputfilepath + inputstr + "gaussian" + F + "_d0=" + to_string(d0) + "_DFT_" + inputname + "_Magt.png";
  imwrite(name, outputMat1);
}
void Gaussian1D(Mat inputMat1, Mat inputMat2, double sig, int xy, int mode, double amp, int d0, string inputname, string inputstr)
{
  string outputfilepath = "../data/output/";
  string name;
  string F;
  int M = inputMat1.rows;
  int N = inputMat1.cols;
  Mat outputMat1(M, N, CV_64FC1);
  Mat display1(M, N, CV_8UC1);
  complex<double> G = 0;
  double duv1;
  double duv2;
  for (int u = 0; u < M; u++)
  {
    for (int v = 0; v < N; v++)
    {
      complex<double> temp = {inputMat1.at<double>(u, v), inputMat2.at<double>(u, v)};
      double dis = sqrt(pow(u - (M / 2.0), 2) + pow(v - (N / 2.0), 2)); //,pow(u-(M/2.0),2)+pow(v-(N/2.0),2)
      if (dis < d0)
      {
        G = 1;
      }
      else if (dis > d0)
      {
        if (mode == 0)
        {
          duv1 = (u - (M / 2) + xy);
          duv2 = (u - (M / 2) - xy);
          F = "Y";
        }
        else if (mode == 1)
        {
          duv1 = (v - (N / 2) + xy);
          duv2 = (v - (N / 2) - xy);
          F = "X";
        }
        G = 1 - ((exp((duv1 * duv1) / ((-2) * pow(sig, 2)))) * amp); //((sig * sqrt(2 * M_PI)) *
        G = G * (1 - ((exp((duv2 * duv2) / ((-2) * pow(sig, 2)))) * amp));
      }

      temp = temp * G;
      display1.at<uchar>(u, v) = G.real() * 255;
      inputMat1.at<double>(u, v) = temp.real();
      inputMat2.at<double>(u, v) = temp.imag();
      outputMat1.at<double>(u, v) = sqrt(pow(temp.real(), 2) + pow(temp.imag(), 2));
    }
  }
  powerLaw(outputMat1, 0.3);
  outputMat1.convertTo(outputMat1, CV_8UC1, 255, 0);
  name = outputfilepath + inputstr + "gaussian1D_" + F + "_sig=" + to_string(sig) + "_DFT_" + inputname + ".png";
  imwrite(name, display1);
  name = outputfilepath + inputstr + "gaussian1D_" + F + "_sig=" + to_string(sig) + "_DFT_" + inputname + "_Magt.png";
  imwrite(name, outputMat1);
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
  Mat kirby_org(512, 512, CV_8UC1);
  Mat flower_org(512, 512, CV_8UC1);
  Mat temp(512, 512, CV_64FC1);
  Mat temp_ln(512, 512, CV_64FC1);
  Mat test(512, 512, CV_64FC1);
  Mat temp_dl(512, 512, CV_64FC1);
  Mat DFT_Real(512, 512, CV_64FC1);
  Mat DFT_Imag(512, 512, CV_64FC1);
  Mat IDFT_Real(512, 512, CV_64FC1);
  Mat PIC_float(512, 512, CV_64FC1);
  Mat out(512, 512, CV_8UC1);
  Mat DFT_Magt(512, 512, CV_64FC1);
  Mat filter_Magt(512, 512, CV_64FC1);

  //-------讀取檔案區------------------------------------------------
  Read_Raw(inputfilepath + "kirby512.raw", rawfilebig, 512, 512);
  for (int i = 0; i < kirby_org.rows; i++)
  {
    for (int j = 0; j < kirby_org.cols; j++)
      kirby_org.at<uchar>(i, j) = rawfilebig[i * kirby_org.cols + j];
  }
  Read_Raw(inputfilepath + "motion_flower.raw", rawfilebig, 512, 512);
  for (int i = 0; i < flower_org.rows; i++)
  {
    for (int j = 0; j < flower_org.cols; j++)
    {
      flower_org.at<uchar>(i, j) = rawfilebig[i * flower_org.cols + j];
    }
  }

  imwrite(outputfilepath + "kirby_org.png", kirby_org);
  imwrite(outputfilepath + "flower_org.png", flower_org);
  while (inputstring != "quit")
  {
    // printf("please enter the question number \n enter quit to exit\n");
    // printf("menu \n 6-1 (will show the best img i think so) \n");
    // printf(" 6-2a-05  6-2a-25  6-2a-125 \n 6-2b-05  6-2b-25  6-2b-125\n");
    cin >> inputstring;
    if (inputstring == "7-1b")
    {
      imshow("kirby", kirby_org);
      waitKey(0);
      destroyAllWindows();
      kirby_org.convertTo(temp, CV_64FC1, (1 / 255.0), 0);
      DFT_TRANS(temp, DFT_Real, DFT_Imag, inputstring);
      BandRejectFilter(DFT_Real, DFT_Imag, 83, 7, inputstring);
      IDFT_TRANS(DFT_Real, DFT_Imag, test, inputstring);
      imshow("temp", temp);
      imshow("test", test);
      waitKey(0);
      destroyAllWindows();
    }
    else if (inputstring == "7-2")
    {

    }
    else if (inputstring == "7-1a")
    {
      imshow("kirby", kirby_org);
      waitKey(0);
      destroyAllWindows();
      kirby_org.convertTo(temp, CV_64FC1, (1 / 255.0), 0);
      DFT_TRANS(temp, DFT_Real, DFT_Imag, inputstring);
      Gaussian1D(DFT_Real, DFT_Imag, 3, 16, 0, 0.8, 70, "filter1", inputstring);
      Gaussian1D(DFT_Real, DFT_Imag, 3, 17, 0, 0.85, 40, "filter2", inputstring);
      Gaussian1D(DFT_Real, DFT_Imag, 2, 81, 1, 0.94, 50, "filter3", inputstring);
      //Gaussian1D(DFT_Real,DFT_Imag,4,82,1,0.8,100);
      NotchFilter(DFT_Real, DFT_Imag, 40, 272 - 256, 337 - 256, "filter4", inputstring);
      IDFT_TRANS(DFT_Real, DFT_Imag, test, inputstring);
      imshow("temp", temp);
      imshow("test", test);
      test.convertTo(test, CV_8UC1, 255, 0);
      imwrite(outputfilepath + "kirby_notchfilter.png", test);
      waitKey(0);
      destroyAllWindows();
    }
  }
  return 0;
}
