
function allGreedyVsTwoStepPlans(nSegments) {
    // for a layer made of nSegments
    // create all possible sequences of layers with hole counts: 1-2-1
    // where the best decision of L vs R at layer 2 differs between greedy and 2-step
    let plans2 = [];
    let plans3 = [];
    for (let hole1 = 1; hole1 < nSegments-1; hole1++) {
        // hole1 cannot be at extreme left/right otherwise no flanking holes possible
        // let curPlans = [];
        for (let hole2L = 0; hole2L < nSegments; hole2L++) {
            if (hole2L > hole1) continue; // left hole must be left of hole1
            for (let hole2R = 0; hole2R < nSegments; hole2R++) {
                if (hole2R < hole1) continue; // right hole must be right of hole1
                if (hole2L === hole2R) continue; // holes must be distinct
                for (let hole3 = 0; hole3 < nSegments; hole3++) {
                    let oneStepPathL = Math.abs(hole1 - hole2L);
                    let oneStepPathR = Math.abs(hole1 - hole2R);
                    let twoStepPathL = oneStepPathL + Math.abs(hole2L - hole3);
                    let twoStepPathR = oneStepPathR + Math.abs(hole2R - hole3);

                    let oneStepChoice = (oneStepPathL < oneStepPathR) ? 'L' : ((oneStepPathL === oneStepPathR) ? 'M' : 'R');
                    let twoStepChoice = (twoStepPathL < twoStepPathR) ? 'L' : ((twoStepPathL === twoStepPathR) ? 'M' : 'R');

                    let curPlan = {
                        hole_locations: [hole1, [hole2L, hole2R], hole3],
                        oneStepChoice: oneStepChoice,
                        multiStepChoice: twoStepChoice,
                        oneStepPathL: oneStepPathL,
                        oneStepPathR: oneStepPathR,
                        multiStepPathL: twoStepPathL,
                        multiStepPathR: twoStepPathR,
                    };
                    plans2.push(curPlan);

                    if (oneStepChoice !== twoStepChoice) continue; // need same choice for 1-step and 2-step
                    let hole3L = hole3;
                    if (hole3L > hole2R) continue; // left hole must be left of right-most hole in 2nd layer
                    for (let hole3R = 0; hole3R < nSegments; hole3R++) {
                        if (hole3R <= hole3L) continue; // right hole must be strictly right of hole3L
                        for (let hole4 = 0; hole4 < nSegments; hole4++) {
                            let threeStepPathLL = Math.abs(hole1 - hole2L) + Math.abs(hole2L - hole3L) + Math.abs(hole3L - hole4);
                            let threeStepPathLR = Math.abs(hole1 - hole2L) + Math.abs(hole2L - hole3R) + Math.abs(hole3R - hole4);
                            let threeStepPathRL = Math.abs(hole1 - hole2R) + Math.abs(hole2R - hole3L) + Math.abs(hole3L - hole4);
                            let threeStepPathRR = Math.abs(hole1 - hole2R) + Math.abs(hole2R - hole3R) + Math.abs(hole3R - hole4);
                            let threeStepPathL = Math.min(threeStepPathLL, threeStepPathLR);
                            let threeStepPathR = Math.min(threeStepPathRL, threeStepPathRR);

                            let threeStepChoice = (threeStepPathL < threeStepPathR) ? 'L' : ((threeStepPathL === threeStepPathR) ? 'M' : 'R');

                            // need 3-step choice to differ from 1-step and 2-step choice
                            // if (threeStepChoice === oneStepChoice) continue;

                            let curPlanExtended = {
                                hole_locations: [hole1, [hole2L, hole2R], [hole3L, hole3R], hole4],
                                oneStepChoice: oneStepChoice,
                                twoStepChoice: twoStepChoice,
                                multiStepChoice: threeStepChoice,
                                oneStepPathL: oneStepPathL,
                                oneStepPathR: oneStepPathR,
                                twoStepPathL: twoStepPathL,
                                twoStepPathR: twoStepPathR,
                                multiStepPathL: threeStepPathL,
                                multiStepPathR: threeStepPathR,
                            };
                            plans3.push(curPlanExtended);
                        }
                    }
                }
            }
        }
        // plans.push(curPlans);
    }
    return [plans2, plans3];
}

function savePlansToJSON(E) {
  let plans = allGreedyVsTwoStepPlans(E.params.nSegments);
  let jsonString = JSON.stringify({plans: plans}, null, 2); // Pretty-print with 2-space indent

  // Create a Blob from the JSON string
  let blob = new Blob([jsonString], { type: 'application/json' });

  // Create a temporary download link
  let url = URL.createObjectURL(blob);
  let a = document.createElement('a');
  a.href = url;
  let saveName = 'plans';
  a.download = `${saveName}.json`;
  a.click();

  // Clean up the URL object
  URL.revokeObjectURL(url);
}
