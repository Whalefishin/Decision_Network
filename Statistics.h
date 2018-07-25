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

inline double rand_normal(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

// double absValue(double x){
//   if (x < 0){
//     x = -x;
//   }
//   return x;
// }
