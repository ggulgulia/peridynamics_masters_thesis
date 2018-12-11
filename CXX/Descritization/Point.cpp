#include "Point.hpp"

//default constructor

PD::Point::Point():m_dimension(0)
{//empty constructor body
} 

//1D constructor
PD::Point::Point(double x):m_dimension(1)
{
    std::shared_ptr<double> tempX(new double(x));
    m_point.push_back(tempX);
}
                //2D Constructor
PD::Point::Point(double x, double y): m_dimension(2)
{
   std::shared_ptr<double> tempX(new double(x));
   std::shared_ptr<double> tempY(new double(y));
   m_point.push_back(tempX);
   m_point.push_back(tempY);

}

//3D Constructor
PD::Point::Point(double x, double y, double z): m_dimension(3)
{
   std::shared_ptr<double> tempX(new double(x));
   std::shared_ptr<double> tempY(new double(y));
   std::shared_ptr<double> tempZ(new double(z));
   m_point.push_back(tempX);
   m_point.push_back(tempY);
   m_point.push_back(tempZ);
}

//TODO implement move constructor
int PD::Point::get_dimension() const { return m_dimension;}
