# Active Beam Tracking with Reconfigurable Intelligent Surface
This repository contains the code for the paper "Active Beam Tracking with Reconfigurable Intelligent Surface"

## [For Python codes]
Required environments: python==3.6, tensorflow==1.15.0, numpy==1.19.2 (Also valid: python==3.9, tensorflow==2.x)

Please generate Fig. 6 before you generate Fig. 5 in order to understand the code better.

## [For plotting Fig. 6]
Please first train each architecture, then test its performance with the generated 'params' file.

Please copy all the generated .mat files into the folder named 'plotting' to run the MATLAB code.

## [For plotting Fig. 5]
Please first train the two approaches using the programs in plotting Fig. 6, then copy the generated 'params' file into their corresponding folder in Fig. 5.

Please only generate the 'Rician_channel' folder once and use the same generated channel statistics to test two approaches (Notice the 'isTesting' option). 

Notice, since the moving trajectory of the UE is randomly generated, there is no guarantee that Fig. 5 can be reproduced exactly the same.

## Questions
If you have any questions, please feel free to reach me at: h.han.working@gmail.com
