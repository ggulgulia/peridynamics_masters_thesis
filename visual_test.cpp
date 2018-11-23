#include <iostream>
#include "visual.hpp"

int main(){

    size_t numX(10), numY(10),numZ(10);
    double length(3);
    double dx, dy, dz;
    size_t gridSize(numX*numY*numZ);

    std::vector<double> X(gridSize,0), Y(gridSize,0), Z(gridSize,0);
    std::string file_name("test");
    dx = length/(numX );
    dy = length/(numY );
    //dz = length/(numZ );
    dz = 0.1;

    for (size_t i = 0; i < numZ; ++i) {
        for (size_t j = 0; j < numY; ++j) {
            for(size_t k=0; k < numX; ++k){

                X[k + j*numY + i*numY*numY] = k*dx;
                Y[k + j*numY + i*numY*numY] = j*dy;
                Z[k + j*numY + i*numY*numY] = i*dz;
            }
        } 
    }

    VTK_Writer vtk_object(X,Y,Z, gridSize, numX, numY, numZ, file_name);
    vtk_object.write_to_file();

    return 0;
}
