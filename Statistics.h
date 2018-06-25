#pragma once

#include <math.h>
#include <iostream>
#include <vector>

using namespace std;

inline double mean(vector<double> data){
  double sum=0;
  double size = data.size();
  for (int i=0;i<data.size();i++){
    sum += data[i];
  }
  return sum/size;
}

inline double standardDeviation(vector<double> data){
  double m = mean(data);
  double sum = 0;
  double size = data.size();

  for (int i=0;i<data.size();i++){
    sum += pow(data[i]-m,2);
  }

  if (size==0){
    return 0;
  }

  return sqrt(sum/size);
};

inline double CoefficientVariation(vector<double> data){
  double m = mean(data);
  double sd = standardDeviation(data);
  return (sd/m);
}

inline double max(double x, double y){
  if (x>y){
    return x;
  }
  return y;
}

inline int mod(int number, int n){
  if (number >=n){
    number -= n;
  }
  else if (number <0){
    number +=n;
  }
  return number;
}

inline double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

template <typename T>
bool contains(vector<T> list, T item){
  for (int i=0;i<list.size();i++){
    if (list[i] == item){
      return true;
    }
  }
  return false;
}

// double absValue(double x){
//   if (x < 0){
//     x = -x;
//   }
//   return x;
// }
