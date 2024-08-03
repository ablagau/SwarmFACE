
# SwarmFACE
[![DOI](https://zenodo.org/badge/548999549.svg)](https://zenodo.org/badge/latestdoi/548999549)
[![docs](https://readthedocs.org/projects/swarmface/badge/?version=latest)](http://swarmface.readthedocs.io/)

**Overview**

The _**SwarmFACE**_ package serves the exploration of field-aligned currents 
(FACs) system based on the magnetic field measurements supplied by the Swarm 
satellites. Improvements of well-established techniques as well as novel 
single- and multi-satellite methods are implemented to extend 
the characterization of FAC systems beyond the Swarm official Level-2 FAC 
product. The package further provides an useful utility to find intervals 
when Swarm forms a close configuration above the auroral oval (AO). In 
addition, for each AO crossing, a series of FAC quality indicators related to
the FAC methods’ underlying assumptions (i.e. FAC sheet planarity and 
orientation, correlation of magnetic field perturbations recorded by the 
lower Swarm satellites), can be estimated.

**Installation**

Clone the SwarmFACE package on your computer and 
go to the newly made SwarmFACE directory

    git clone https://github.com/ablagau/SwarmFACE
    cd SwarmFACE

After you activate the python environment you 
can use pip:

`pip install ./`

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
and two for estimating the FAC quality indices. To run, e.g., the 
function that estimates the FAC density by the 
single-satellite method, write:

    dtime_beg = '2014-05-04T17:48:00'
    dtime_end = '2014-05-04T17:55:00'
    sat = ['C'] 
    j_df, dat_df, param = j1sat(dtime_beg, dtime_end, sat)

For each high-level function, there is an associated 
jupyter notebook in the _notebooks_ directory that illustrates 
its application. 

The application and performance of the FAC estimation methods
were extensively discussed in 

Blagau, A., and Vogt, J. (2019). Multipoint field-aligned current
estimates with Swarm. J. Geophys. Res. (Space Phys. 124, 6869–6895.
doi:10.1029/2018JA026439

Vogt, J., Blagau, A., and Pick, L. (2020). Robust adaptive spacecraft
array derivative analysis. Earth Space Sci. 7, e00953. 
doi:10.1029/2019EA000953


**Acknowledging and citing SwarmFACE**

If you use SwarmFACE for scientific work or research presented 
in a publication, please cite the SwarmFACE paper:

Blagau, A., and Vogt, J., (2023), SwarmFACE: A Python package for 
field-aligned currents exploration with Swarm. Front. Astron. 
Space Sci. 9:1077845. doi: 10.3389/fspas.2022.1077845

Additionally, consider to add in the methods or acknowledgements 
section the following: "This research has made use of SwarmFACE v?.?.?, 
an open-source and free Python package that serves the 
exploration of field-aligned currents system based on 
Swarm observation (Zenodo: https://doi.org/10.5281/zenodo.7361438)." 


**Copyright**

© 2022 Adrian Blagau and Joachim Vogt. SwarmFACE is an 
open-access code distributed under the terms of the 
Creative Commons Attribution License (CC BY). 
The use, distribution or reproduction in other forums is 
permitted, provided the original author(s) and the copyright 
owner(s) are credited and that the publication
describing the code (see above) is cited, in accordance 
with accepted academic practice.

The software is provided "as is", without warranty of any kind, 
express or implied, including but not limited to the warranties 
of merchantability, fitness for a particular purpose and non 
infringement. In no event shall the authors or copyright 
holders be liable for any claim, damages or other liability, 
whether in an action of contract, tort or otherwise, 
arising from, out of or in connection with the software or 
the use or other dealings in the software.
