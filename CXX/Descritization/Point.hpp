#ifndef PD_POINT_H
#define PD_POINT_H 

#include <vector>
#include <memory>

namespace PD
{

        class Point{
            private:
                std::vector<std::shared_ptr<double>> m_point;
                int m_dimension;
            public:
                //default constructor
                Point();
                //1D constructor
                Point(double x);                
                //2D Constructor
                Point(double x, double y);
               //3D Constructor
               Point(double x, double y, double z);

                Point(const Point& p) = delete;
                //TODO implement move constructor
                Point(Point&& p);
                int get_dimension() const;
        };
}
#endif /* ifndef PD_MESH_H */
