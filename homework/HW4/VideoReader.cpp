#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//使用方法
/*
Put street.avi in your project.
Set the link with opencv.
*/

int main() {

    VideoCapture cap("street.avi");
 
    if (!cap.isOpened()) {
        cout << "Error opening video" << endl;
        return -1;
    }

    while (1) {
        Mat frame;
        //下方程式碼會依序獲得每一帧
        //請根據上課所學修改此段
        cap >> frame;

        //若影片結束跳出迴圈
        if (frame.empty())
            break;

        
        imshow("Frame", frame);

        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}