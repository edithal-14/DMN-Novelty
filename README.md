# Description
Exploring Dynamic memory networks for novelty detection

# Link to paper
* https://aclanthology.org/2022.paclic-1.58.pdf
* This paper was accepted in 36th Pacific Asia Conference on Language, Information and Computation (PACLIC 36) which took place from October 20-22, 2022, in a fully virtual format in the Philippines

# References
* This code is based on the original DMN+ paper: https://arxiv.org/abs/1603.01417
* This code is a modified version of the DMN+ implementation by dandelin: https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch

# Datasets used
* DLND
* DLND 2.0
* STE (Stack Exchange)
* APWSJ
* WEBIS-CPC 11

### How to create STE dataset

* You can download the 06 June 2022 Stack Exchange data dump from here: https://archive.org/download/stackexchange_20220606
* This is the version we used in our paper
* You will have to download the .7z file for each topic individually
* Even though the paper states we used 50 topics, we actually ended up using only the following 42 topics due to time limitation:
  - 3dprinting
  - academia
  - beer
  - blender
  - boardgames
  - cicvicrm
  - coffee
  - crypto
  - cs
  - cstheory
  - dsp
  - emacs
  - engineering
  - expressionengine
  - fitness
  - german
  - hermeneutics
  - islam
  - judaism
  - magento
  - matheducators
  - mathematica
  - meta.mathoverflow.net
  - meta.serverfault.com
  - meta.stackoverflow.net
  - meta.superuser
  - money
  - mythology
  - pets
  - philosophy
  - robotics
  - rpg
  - scicomp
  - sharepoint
  - skeptics
  - softwarerecs
  - space
  - sqa
  - tridion
  - webmasters
  - windowsphone
  - woodworking


* README file to understand the xml files in the dataset: https://ia601502.us.archive.org/24/items/stackexchange_20220606/readme.txt

* This is the link to the script which is used to create the .json files from the Stack Exchange Data Dump: https://github.com/Lab41/pythia/blob/master/src/data/stackexchange/stack_exchange_parse.py
  - This script was developed by Lab41 as part of their Pythia project to create a Novelty detection dataset from Stack Exchange dataset.

* The section 3 of the paper (https://aclanthology.org/2022.paclic-1.58.pdf) talks about the logic behind the script but here is a more detailed explanation
  - Summary of the script (see iter_clusters() function for more details)
  - We give a file containing a list of topics to the script, for each topic it does the following, I am taking **space** as an example topic here.
  - Download the dataset from the URL:  https://archive.org/download/stackexchange_20220606/<topic>.meta.stackexchange.com.7z and extract it
    - For example, download the url: https://archive.org/download/stackexchange_20220606/space.meta.stackexchange.com.7z
  - First look at the PostLinks.xml (https://ia601502.us.archive.org/view_archive.php?archive=/24/items/stackexchange_20220606/space.stackexchange.com.7z&file=PostLinks.xml) file
    - Only consider entries which are related (LinkTypeId="1") or duplicate (LinkTypeId="3"). Meaning of LinkTypeId is given in the above README
    - For example see row Id="556" where PostId="447", RelatedPostId="78" and LinkTypeId="3"
    - PostId is target question id and RelatedPostId is source question id
  - For each entry, search PostHistory.xml (https://ia801502.us.archive.org/view_archive.php?archive=/24/items/stackexchange_20220606/space.stackexchange.com.7z&file=PostHistory.xml) for entries where PostId = target question id
    - If there is an entry with PostHistoryTypeId="10" and Comment="101" then the target question was closed against the source question.
    - For example check entry with row Id="1411" where PostId="447" which is our target question id
    - If there is such an entry then mark the pair as Non-Novel otherwise Novel
  - See Posts.xml and get the body of the target question and source question by searching for entry with the correct Id
