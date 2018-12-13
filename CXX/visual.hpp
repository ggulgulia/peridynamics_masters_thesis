
#ifndef VISUAL_H
#define VISUAL_H

#include <fstream>
#include <vector>
#include <string>

class VTK_Writer{

    private:
        std::vector<double> m_Xcoordinates;
        std::vector<double> m_Ycoordinates;
        std::vector<double> m_Zcoodrinates;

        size_t m_gridSize;
        size_t m_numX, m_numY, m_numZ;
        std::string m_file_name;
    
    public:
        VTK_Writer() = delete;
        VTK_Writer(const VTK_Writer&) = delete;

        //constructor
        VTK_Writer(std::vector<double> X_coords, 
                std::vector<double> Y_coords,
                std::vector<double> Z_coords, 
                const size_t gridSize, 
                const size_t numX,
                const size_t numY,
                const size_t numZ,
                std::string file_name);

        void write_to_file(); 
        void write_header(std::ofstream& file);
};

#endif
