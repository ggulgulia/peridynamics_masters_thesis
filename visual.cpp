#include "visual.hpp"


        //constructor
VTK_Writer::VTK_Writer(std::vector<double> X_coords, 
                std::vector<double> Y_coords,
                std::vector<double> Z_coords, 
                const size_t gridSize, 
                const size_t numX,
                const size_t numY,
                const size_t numZ,
                std::string file_name):
            m_Xcoordinates(X_coords), 
            m_Ycoordinates(Y_coords), 
            m_Zcoodrinates(Z_coords),
            m_gridSize(gridSize), 
            m_numX(numX),
            m_numY(numY),
            m_numZ(numZ),
            m_file_name(file_name.append(".vtk")){

                //empty body
            }

        void VTK_Writer::write_to_file() {
            std::ofstream myFile;
            myFile.open(m_file_name);
            write_header(myFile);

            for (size_t i = 0; i < m_gridSize; ++i) {
                myFile << m_Xcoordinates[i] << " " << m_Ycoordinates[i] << " " << m_Zcoodrinates[i] << "\n";                
            }

            myFile.close();
        }

        void VTK_Writer::write_header(std::ofstream& file){
            file << "# vtk DataFile Version 2.0\n";
            file << "generated for peridym project\n";
            file << "ASCII\n\n";
            file << "DATASET STRUCTURED_GRID\n";
            file << "DIMENSIONS  " << m_numX << " " << m_numY<< " " << m_numZ << "\n";
            file << "POINTS " << m_gridSize << " float\n";
            file << "\n";
        }
