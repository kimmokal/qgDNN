#!/bin/bash

# Path to the working directory
WORK_DIR=/work/kimmokal/qgDNN

# Path to the original ntuples and the output directory
DATA_DIR=/work/data/QCD_jetTuples_Pythia8_PU_Moriond17/merged/
OUT_DIR=$WORK_DIR/data/binned/

# Define the seven eta, pT bins. The quark and gluon jets are also separated at this step plus some quality selections.
CUT1_Q="(jetPt>30)&&(jetPt<100)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT2_Q="(jetPt>100)&&(jetPt<300)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT3_Q="(jetPt>300)&&(jetPt<1000)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT4_Q="(jetPt>1000)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT5_Q="(jetPt>30)&&(jetPt<100)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT6_Q="(jetPt>100)&&(jetPt<300)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT7_Q="(jetPt>300)&&(jetPt<1000)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysUDS==1)"
CUT1_G="(jetPt>30)&&(jetPt<100)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT2_G="(jetPt>100)&&(jetPt<300)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT3_G="(jetPt>300)&&(jetPt<1000)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT4_G="(jetPt>1000)&&(fabs(jetEta)<1.3)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT5_G="(jetPt>30)&&(jetPt<100)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT6_G="(jetPt>100)&&(jetPt<300)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"
CUT7_G="(jetPt>300)&&(jetPt<1000)&&(fabs(jetEta)>1.3)&&(fabs(jetEta)<2.5)&&(jetTightID==1)&&(jetQGl>=0)&&(isPhysG==1)"

# Define the corresponding filenames
F1_Q="quark_eta1.3_pt30to100"
F2_Q="quark_eta1.3_pt100to300"
F3_Q="quark_eta1.3_pt300to1000"
F4_Q="quark_eta1.3_pt1000"
F5_Q="quark_eta2.5_pt30to100"
F6_Q="quark_eta2.5_pt100to300"
F7_Q="quark_eta2.5_pt300to1000"
F1_G="gluon_eta1.3_pt30to100"
F2_G="gluon_eta1.3_pt100to300"
F3_G="gluon_eta1.3_pt300to1000"
F4_G="gluon_eta1.3_pt1000"
F5_G="gluon_eta2.5_pt30to100"
F6_G="gluon_eta2.5_pt100to300"
F7_G="gluon_eta2.5_pt300to1000"

CUTS=( $CUT1_Q $CUT2_Q $CUT3_Q $CUT4_Q $CUT5_Q $CUT6_Q $CUT7_Q $CUT1_G $CUT2_G $CUT3_G $CUT4_G $CUT5_G $CUT6_G $CUT7_G )
FNAMES=( $F1_Q $F2_Q $F3_Q $F4_Q $F5_Q $F6_Q $F7_Q $F1_G $F2_G $F3_G $F4_G $F5_G $F6_G $F7_G )

# Loop over the seven eta, pT bins and q/g
for i in {0..13}
do
    # The counter is used to keep track of the number of files in the 'merged' directory
    COUNTER=1
    # Create an array for the output files for later merging
    OUT_FILES=()

    for FILE in $DATA_DIR/*
    do
        OUT_NAME="${FNAMES[i]}_${COUNTER}.root"
        echo "Processing: ${OUT_NAME}"

        # Append the file to the list
        OUT_FILES=( "${OUT_FILES[@]}" "${OUT_DIR}${OUT_NAME}" )

        # Use rooteventselector available in ROOT 6
        rooteventselector -s ${CUTS[i]} $FILE:demo/jetTree $OUT_DIR/$OUT_NAME

        COUNTER=$((COUNTER+1))
    done

    # Use hadd to merge the files
    hadd "$OUT_DIR${FNAMES[i]}.root" ${OUT_FILES[@]}

    # Remove the original files
    for FILE in ${OUT_FILES[@]}
    do
        rm $FILE
    done

    echo "Processed: ${FNAMES[i]}"
done
