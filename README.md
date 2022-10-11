**Overview**

The _**SwarmFACE**_ package serves the exploration of field-aligned currents 
(FACs) system based on the magnetic field measurements supplied by the Swarm 
satellites. Improvements of well-established techniques as well as novel 
single- and multi-satellite methods are implemented to extend 
the characterization of FAC systems beyond the Swarm official Level-2 FAC 
product. The package further provides an useful utility to find intervals 
when Swarm forms a close configuration above the auroral oval (AO). In 
addition, for each AO crossing, a series of FAC quality indicators related to the FAC methods’ 
underlying assumptions (i.e. FAC sheet planarity and orientation, correlation 
of magnetic field perturbations recorded by the lower Swarm satellites), can 
be estimated.

**Installation**

Clone the SwarmFACE package on your computer and 
go to the newly made SwarmFACE directory

    git clone https://github.com/ablagau/SwarmFACE
    cd SwarmFACE

After you activate the python environment you 
can use pip3:

`pip3 install ./`

**Dependencies**

* viresclient
* numpy
* pandas
* scipy
* jupyter
* matplotlib

**Quick start**

To import the package write:

    from SwarmFACE import *

There are five high-level functions to estimate the FAC 
density, one to find Swarm conjunctions above the auroral oval, 
and two for estimating the FAC quality indices. To run, e.g., the function that estimates the FAC density by the 
single-satellite method, write:

    dtime_beg = '2014-05-04T17:48:00'
    dtime_end = '2014-05-04T17:55:00'
    sat = ['C'] 
    j_df, dat_df, param = j1sat(dtime_beg, dtime_end, sat)

For each high-level function, there is an associated 
jupyter notebook that illustrates its application _notebooks_ 
directory.

**Copyright**

© 2022 Adrian Blagau and Joachim Vogt. SwarmFACE is an 
open-access code distributed under the terms of the 
Creative Commons Attribution License (CC BY). 
The use, distribution or reproduction in other forums is 
permitted, provided the original author(s) and the copyright 
owner(s) are credited and that the following publication 
describing the code is cited, in accordance with accepted 
academic practice:

Blagau, A., J. Vogt, (2022), SwarmFACE: a Python Package for 
Field-Aligned Currents Exploration with Swarm, Front. Astron. 
Space Sci., under review.

The software is provided "as is", without warranty of any kind, 
express or implied, including but not limited to the warranties 
of merchantability, fitness for a particular purpose and non 
infringement. In no event shall the authors or copyright 
holders be liable for any claim, damages or other liability, 
whether in an action of contract, tort or otherwise, 
arising from, out of or in connection with the software or 
the use or other dealings in the software.