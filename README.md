# CPMclass

This repository includes:
1. A c++ class to interpret CPM caffe deploy (.prototxt) and learnt model (.caffemodel) files, obtaining person skeletons. This class is defined by the files cpm_c++.cpp and cpm_c++.h

2. A caffe python layer (parts_extractor.py with config file) whose input is a person image blob and its outputs are nine body parts images blobs corresponging to: head, upper right arm, lower right arm, upper laeft arm, lower left arm, upper right leg, lower right leg, upper left leg, and lower left leg. The deploy and learnt model files for the CPM must be given.
 
<br />

# Citation:

Please cite CPM c++ class in your publications if it helps your research:

Gómez-Silva, M. J., Armingol, J. M., & de la Escalera, A. (2018). Multi-Object Tracking Errors Minimisation by Visual Similarity and Human Joints Detection. In 8th International Conference on Imaging for Crime Detection and Prevention (ICDP 2017) (pp. 25-30).

@inproceedings{gomez2016multi,<br />
  title={Multi-Object Tracking Errors Minimisation by Visual Similarity and Human Joints Detection},<br />
  author={G{\'o}mez-Silva, Mar{\'i}a Jos{\'e} and Armingol, Jos{\'e} Mar{\'i}a and de la Escalera, Arturo},<br />
  booktitle={8th International Conference on Imaging for Crime Detection and Prevention (ICDP 2017)},<br />
  pages={25--30},<br />
  year={2018},<br />
  organization={IET}<br />
}
 
 <br />

Please cite parts_extractor python layer in your publications if it helps your research:

Gómez-Silva, M. J., Armingol, J. M., & de la Escalera, A. (2018). Deep Parts Similarity Learning for Person Re-Identification.  In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018) - Volume 5: VISAPP, pages 419-428.

@inproceedings{gomez2017deep, <br />
 title={Deep Parts Similarity Learning for Person Re-Identification.}, <br />
 author={G{'o}mez-Silva, Mar{'\i}a Jos{'e} and Armingol, Jos{'e} Mar{'\i}a and de la Escalera, Arturo}, <br />
 booktitle={In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018)}, <br />
 pages={419--428}, <br />
 year={2018} <br />
}
