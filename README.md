Peridynamics offers a new methodology for modelling solid mechanics problems using the idea of interactions of a point with the neigboring points and is a different framework compared to the classical continuum mechanics that is being used most widely. 

This project demonstrates the usage of Perdiynamics in an academic setting that helps to understand the usage of peridynamics theory of continuum mechanics which is more robust than classical continuum mechanics to deal with problems involving discontinuiteis and inclusions especially in fracture mechanics. 
 
## The code has the following features:

#### Two choices of material model namely 
* Constitutive material model (look at the file peridynamics_stiffness.py)
* Correspondence material model ((look at the file peridynamics_correspondence.py)

#### Choices of influence functions namely 
* Standard gaussian influence function (Omega_1)
* Narrow gaussian influence function (Omega_2)
* Unit-Step influence function (Omega_3)
* Parabolic influcene function (Omega_4)
* Peridigm-like parabolic function (Omega_5)

![](influenceFunctions.png)

#### Choices of solution  : static and dynamic (see image/animation below)
* static bending of a 2D and 3D plate (only a 2D example shown below)
![](2DBending.png)

* dynamic fracture of 2D plate
![](explicitFracture.gif)

