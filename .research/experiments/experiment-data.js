const COLORS = {
  blue: "#1677c8", orange: "#d87824", green: "#258b50", red: "#c43d45",
  purple: "#7755ba", teal: "#147f80", gray: "#718096"
};

const MNIST_BASE = {
  title: "MNIST MLP · 256×2",
  note: "Shared Step 1 architecture used by the relaxation and continual-learning studies.",
  settings: [
    ["Architecture", "784 → 256 → 256 → 10; ReLU; biases"],
    ["Learned parameters", "269,322"],
    ["Objective", "One-hot half squared error"],
    ["BP update", "Adam, lr=0.001; one update per batch"],
    ["PC state update", "Latent SGD, lr=0.01; update_p_at=last"],
    ["Batch size", "500"]
  ],
  flow: ["784 pixels", "256 + ReLU", "256 + ReLU", "10 logits"],
  latent: ["x₁ · hidden state", "x₂ · hidden state"]
};

const MNIST_WIDE = {
  title: "MNIST MLP · width intervention",
  note: "The hidden width changes while optimizer, stream and evaluation remain frozen.",
  settings: [
    ["Control", "784 → 256 → 256 → 10; 269,322 parameters"],
    ["Intervention", "784 → 512 → 512 → 10; 669,706 parameters"],
    ["Parameter multiplier", "2.487×"],
    ["Learning rule", "BP and PC T=5"],
    ["Optimizer", "Adam, lr=0.001"],
    ["PC state update", "Latent SGD, lr=0.01"]
  ],
  flow: ["784 pixels", "256 or 512", "256 or 512", "10 logits"],
  latent: ["x₁ · matched state", "x₂ · matched state"]
};

const MNIST_DEEP = {
  title: "MNIST MLP · depth intervention",
  note: "Depth changes independently; T=5 is only the propagation minimum for four PC layers.",
  settings: [
    ["Control", "784 → 256 → 256 → 10; 269,322 parameters"],
    ["Intervention", "784 → 256×4 → 10; 400,906 parameters"],
    ["Learning rule", "BP and PC T=5"],
    ["Optimizer", "Adam, lr=0.001"],
    ["Training budget", "Frozen across depth"],
    ["Inference margin", "Minimal for deep PC; not depth-matched"]
  ],
  flow: ["784 pixels", "2 or 4 hidden layers", "10 logits"],
  latent: ["x₁ … x₄ · inferred states"]
};

const CIFAR_MODEL = {
  title: "CIFAR-10 ConvNet",
  note: "Source-derived convolutional architecture used for static reproduction and SplitCIFAR-10.",
  settings: [
    ["Architecture", "Conv 3→64 → Conv 64→128 → FC 8192→512→10"],
    ["Learned parameters", "4,275,402"],
    ["Objective", "One-hot half squared error"],
    ["RBP control", "16 Adam updates per batch; weight decay 0.01"],
    ["PC inference", "T=16; latent SGD 0.5; update_p_at=all"],
    ["Batch size", "200"]
  ],
  flow: ["3×32×32", "Conv1 · 64", "Conv2 · 128", "FC · 512", "10 logits"],
  latent: ["x₁ · conv1", "x₂ · conv2", "x₃ · fc1"]
};

const STATIC_CURVE = {
  type: "line", title: "Static validation learning curve", subtitle: "Mean over paired seeds 7/42/123; epoch 0 is the shared initialization.",
  x: [0,1,2,3,4,5,6,7,8,9,10], xLabel: "epoch", yLabel: "validation accuracy (%)", yMin: 0, yMax: 100,
  series: [
    {name:"BP", color:COLORS.blue, y:[10.15,94.742,96.314,97.008,97.356,97.608,97.842,97.961,97.958,98.025,98.122]},
    {name:"PC T=1", color:COLORS.red, dash:true, y:[10.15,71.983,76.322,78.389,79.942,80.975,81.731,82.458,82.906,83.311,83.581]},
    {name:"PC T=5", color:COLORS.green, y:[10.15,94.764,96.336,97.019,97.364,97.622,97.800,97.961,97.936,98.042,98.108]},
    {name:"PC T=10", color:COLORS.orange, y:[10.15,94.764,96.322,97.031,97.414,97.672,97.828,97.969,97.983,98.094,98.133]},
    {name:"PC T=20", color:COLORS.purple, y:[10.15,94.825,96.264,97.033,97.389,97.614,97.786,97.906,97.933,98.039,98.061]}
  ],
  takeaway: "T=5/10/20 tracks BP after the first epoch. T=1 is a propagation-deficient boundary condition."
};

const SPLIT_CURVE = {
  type: "line", title: "SplitMNIST prequential accuracy by task", subtitle: "Predict-before-update accuracy averaged across paired seeds.",
  x:[1,2,3,4,5], xLabel:"task", yLabel:"prequential accuracy (%)", yMin:0, yMax:100,
  series:[
    {name:"BP",color:COLORS.blue,y:[93.420,61.355,49.756,49.336,44.224]},
    {name:"PC T=1",color:COLORS.red,dash:true,y:[34.540,3.645,3.544,11.807,9.083]},
    {name:"PC T=5",color:COLORS.green,y:[93.417,61.361,45.935,46.124,37.799]},
    {name:"PC T=10",color:COLORS.orange,y:[93.414,62.027,43.126,43.400,37.365]},
    {name:"PC T=20",color:COLORS.purple,y:[93.391,61.787,42.265,36.642,35.540]}
  ],
  takeaway:"Later-task prequential accuracy falls as relaxation increases; lower forgetting is not a free stability gain."
};

const PERMUTED_CURVE = {
  type:"line", title:"Permuted-MNIST prequential accuracy by task", subtitle:"Domain-incremental predict-before-update accuracy averaged across paired seeds.",
  x:[1,2,3,4,5], xLabel:"task", yLabel:"prequential accuracy (%)", yMin:40, yMax:90,
  series:[
    {name:"BP",color:COLORS.blue,y:[87.374,87.357,87.416,86.455,86.172]},
    {name:"PC T=1",color:COLORS.red,dash:true,y:[49.558,50.044,48.544,44.417,44.956]},
    {name:"PC T=5",color:COLORS.green,y:[87.394,87.367,87.526,86.429,86.337]},
    {name:"PC T=10",color:COLORS.orange,y:[87.401,87.430,87.553,86.626,86.633]},
    {name:"PC T=20",color:COLORS.purple,y:[87.509,87.634,87.414,86.681,85.990]}
  ],
  takeaway:"BP and effective PC settings remain tightly clustered. T=1 alone exposes insufficient propagation."
};

const SPLIT_CIFAR_PREQUENTIAL = {
  type:"line", title:"SplitCIFAR-10 prequential accuracy", subtitle:"Predict-before-update accuracy by incoming class-pair task.",
  x:[1,2,3,4,5], xLabel:"task", yLabel:"prequential accuracy (%)", yMin:65, yMax:90,
  series:[
    {name:"Archived RBP",color:COLORS.blue,y:[85.363,74.633,78.108,86.143,85.339]},
    {name:"PC",color:COLORS.orange,y:[80.992,70.969,71.003,77.604,79.727]}
  ],
  takeaway:"PC is below RBP on every incoming task, so reduced forgetting must be read beside weaker acquisition."
};

const SPLIT_CIFAR_RETENTION = {
  type:"line", title:"Retained accuracy after each task boundary", subtitle:"Mean accuracy over tasks seen so far, averaged across paired seeds.",
  x:[1,2,3,4,5], xLabel:"task boundary", yLabel:"seen-task accuracy (%)", yMin:0, yMax:100,
  series:[
    {name:"Archived RBP",color:COLORS.blue,y:[91.400,38.825,27.489,23.258,18.057]},
    {name:"PC",color:COLORS.orange,y:[86.900,37.842,25.022,22.013,16.973]}
  ],
  takeaway:"Both methods collapse as classes accumulate. Neither solves catastrophic forgetting."
};

const EXPERIMENTS = {
  "cifar-figure-4i": {
    title:"Reproduce the paper-defined static CIFAR-10 outcome", short:"CIFAR Figure 4i", status:"Supported narrowly", statusClass:"ok", source:"CIFAR study · H1.E2", proposed:false,
    summary:"A source-faithful baseline that separates implementation validity from broader superiority claims.",
    answer:"All six source-target cells pass, but the result applies only to the archived repeated-update, test-informed Figure 4i protocol.",
    model:CIFAR_MODEL,
    experiment:{title:"Static reproduction protocol",note:"Exact-as-practical rerun of the archived winning configurations.",settings:[["Regime","Static CIFAR-10; 80 training passes"],["Seeds","1482555873, 698841058, 2283198659"],["Comparators","Archived repeated-update BP (RBP) and PC"],["Selection","Source-winning per-method/per-seed LR; best test epoch"],["Gate","Every method-seed cell within ±1.0 pp of source"],["Validity checks","Paired initialization/data checksums; finite metrics; PC descent"]]},
    results:[
      {kind:"achieved",title:"All six reproduction cells passed",metric:"6 / 6",text:"Every RBP and PC best-test value landed within the preregistered ±1 pp source tolerance."},
      {kind:"achieved",title:"Paper-protocol ordering reproduced",metric:"+2.296 pp",text:"Descriptive reproduced PC−RBP mean best accuracy under the archived settings."},
      {kind:"boundary",title:"Higher resource cost",metric:"1.228×",text:"PC synchronized update time relative to RBP, plus about 72.9 MB peak allocated memory."},
      {kind:"non-achieved",title:"Conventional BP superiority not tested",metric:"Not established",text:"The comparator repeats 16 parameter updates per batch and selection is test-informed."}
    ],
    table:{headers:["Evidence","RBP best accuracy","PC best accuracy","PC − RBP"],rows:[["Archived Figure 4i source","69.112%","71.599%","+2.486 pp"],["ROCm reproduction","68.966%","71.262%","+2.296 pp"]]},
    charts:[{type:"line",title:"CIFAR-10 test-accuracy trajectory",subtitle:"Sampled mean trajectory across the 80 archived-protocol passes; full histories remain in the raw artifact.",x:[1,10,20,30,40,50,60,70,80],xLabel:"training pass",yLabel:"test accuracy (%)",yMin:48,yMax:73,series:[{name:"Archived RBP",color:COLORS.blue,y:[61.452,68.687,67.990,67.605,67.364,66.616,66.395,66.269,66.214]},{name:"PC",color:COLORS.orange,y:[50.571,60.388,63.265,65.524,67.548,69.357,70.340,70.850,70.619]}],takeaway:"RBP peaks early; PC learns more slowly and continues improving. This does not compare PC with conventional one-update BP."}],
    claims:[
      {type:"hypothesis",title:"Registered hypothesis",text:"The archived PC and RBP configurations can be reproduced within ±1 pp for every method-seed cell."},
      {type:"verified",title:"Verified evidence",text:"All six tolerance flags, paired checksums, finite-metric checks and PC-descent checks passed."},
      {type:"interpretation",title:"Interpretation",text:"The archived implementation is a valid source-faithful baseline on this stack."},
      {type:"open",title:"Claim ceiling",text:"No leakage-free selection, conventional-BP, continual-learning, biological or general-superiority conclusion follows."}
    ],
    sources:[["Evaluation","../../agents/research/predictive-coding-cifar-baseline/04-evaluation.md"],["Synthesis","../../agents/research/predictive-coding-cifar-baseline/05-synthesis.md"],["Raw run","../../playground/step2-cifar10/runs/paper-reproduction-v1.json"]]
  },

  "static-mnist-relaxation": {
    title:"Find the effective PC inference boundary on static MNIST", short:"Static relaxation sweep", status:"Threshold supported", statusClass:"ok", source:"Relaxation study · H1.E1", proposed:false,
    summary:"Change only PC relaxation depth and observe learning speed, final accuracy and cost.",
    answer:"Five tested steps cross the useful propagation threshold; more relaxation adds cost without a monotonic accuracy benefit.",
    model:MNIST_BASE,
    experiment:{title:"Static relaxation protocol",note:"T is the only intervention.",settings:[["Dataset","MNIST; deterministic 80/20 train/validation split"],["Seeds","7, 42, 123"],["Training","10 epochs; full train/validation/test"],["PC relaxation","T={1,5,10,20}"],["Control","Fresh BP run shared across depths"],["Metrics","Validation trajectory, final test, synchronized runtime, memory"]]},
    results:[
      {kind:"achieved",title:"Effective PC reaches static parity",metric:"98.193%",text:"PC T=5 mean final test accuracy versus BP 98.267%."},
      {kind:"boundary",title:"T=1 is propagation-deficient",metric:"−13.81 pp",text:"Large deficit versus BP is a boundary condition, not ordinary low-budget behavior."},
      {kind:"non-achieved",title:"More steps do not keep helping",metric:"Rejected",text:"T=10 and T=20 do not improve final accuracy over T=5."},
      {kind:"boundary",title:"Runtime rises monotonically",metric:"4.23×",text:"PC T=20 synchronized training-loop time relative to BP."}
    ],
    table:{headers:["Method","Final test","Runtime ratio","Reading"],rows:[["BP","98.267%","1.00×","Control"],["PC T=1","84.457%","1.02×","Boundary failure"],["PC T=5","98.193%","1.78×","Cheapest effective tested"],["PC T=10","98.183%","2.57×","No accuracy gain"],["PC T=20","98.137%","4.23×","No accuracy gain"]]},
    charts:[STATIC_CURVE,{type:"bar",title:"Accuracy and compute across relaxation depth",subtitle:"Final test accuracy and synchronized runtime ratio are shown as separate series.",x:["BP","T=1","T=5","T=10","T=20"],xLabel:"condition",yLabel:"scaled value",yMin:0,yMax:100,series:[{name:"Final accuracy (%)",color:COLORS.blue,y:[98.267,84.457,98.193,98.183,98.137]},{name:"Runtime ratio × 20",color:COLORS.orange,y:[20,20.36,35.5,51.48,84.64]}],takeaway:"Accuracy saturates when propagation becomes effective; compute continues to rise."}],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"Relaxation depth changes learning speed and final performance, with a sufficient depth enabling upstream error propagation."},{type:"verified",title:"Verified observation",text:"T=1 fails; T=5/10/20 cluster near BP throughout the validation trajectory."},{type:"rejected",title:"Rejected component",text:"More relaxation does not monotonically improve accuracy."},{type:"interpretation",title:"Interpretation",text:"T=5 is the cheapest effective tested point, not a global optimum because T=3/4 were not tested."}],
    sources:[["Evaluation","../../agents/research/pc-relaxation-plasticity/04-evaluation.md"],["Synthesis","../../agents/research/pc-relaxation-plasticity/05-synthesis.md"],["Raw run","../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json"]]
  },

  "permuted-mnist-relaxation": {
    title:"Test relaxation under domain-incremental shift", short:"Permuted MNIST", status:"Near parity", statusClass:"ok", source:"Relaxation study · H1.E3", proposed:false,
    summary:"Raise non-stationarity while preserving labels and a one-pass, no-replay protocol.",
    answer:"Effective PC remains BP-like; T=10 is a replication candidate, not evidence of superiority or an optimum.",
    model:MNIST_BASE,
    experiment:{title:"Permuted-MNIST stream",note:"Five domains share labels and a single output head.",settings:[["Regime","Domain incremental"],["Tasks","Identity plus four fixed pixel permutations"],["Training","One pass per task; no replay; no model task ID"],["Seeds","7, 42, 123"],["PC relaxation","T={1,5,10,20}"],["Metrics","Prequential, adaptation, final accuracy, forgetting/BWT"]]},
    results:[{kind:"achieved",title:"Effective PC is BP-like",metric:"91.806%",text:"Best tested PC mean final accuracy occurs at T=10 versus BP 91.544%."},{kind:"boundary",title:"Paired differences have mixed signs",metric:"n=3",text:"The small mean advantage does not establish superiority."},{kind:"non-achieved",title:"No global relaxation optimum",metric:"Not established",text:"T=10 is specific to this finite domain-incremental grid."},{kind:"boundary",title:"T=1 remains invalid as an operating point",metric:"51.008%",text:"The propagation-deficient setting fails across domains."}],
    table:{headers:["Method","Final average","Interpretation"],rows:[["BP","91.544%","Control"],["PC T=1","51.008%","Boundary failure"],["PC T=5","91.326%","Near parity"],["PC T=10","91.806%","Replication candidate"],["PC T=20","91.037%","No added benefit"]]},
    charts:[PERMUTED_CURVE],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"Relaxation depth changes plasticity and stability under domain shift."},{type:"verified",title:"Observation",text:"BP and PC T=5/10/20 remain tightly clustered across all five domains."},{type:"interpretation",title:"Interpretation",text:"T=10 is the best tested PC mean but the effect is small and seed-dependent."},{type:"open",title:"Open verification",text:"Repeat with more seeds before selection or superiority claims."}],
    sources:[["Evaluation","../../agents/research/pc-relaxation-plasticity/04-evaluation.md"],["Synthesis","../../agents/research/pc-relaxation-plasticity/05-synthesis.md"],["Raw run","../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json"]]
  },

  "split-mnist-relaxation": {
    title:"Test whether lower forgetting preserves new-task plasticity", short:"SplitMNIST", status:"Trade-off", statusClass:"warn", source:"Relaxation study · H1.E2", proposed:false,
    summary:"Class-incremental learning exposes the difference between remembering less badly and learning well.",
    answer:"Lower forgetting at larger T co-occurs with weaker acquisition; every method still catastrophically forgets.",
    model:MNIST_BASE,
    experiment:{title:"SplitMNIST stream",note:"Five class pairs compete through one shared ten-way head.",settings:[["Regime","Class incremental"],["Tasks","0/1, 2/3, 4/5, 6/7, 8/9"],["Training","One pass per task; no replay; no model task ID"],["Seeds","7, 42, 123"],["PC relaxation","T={1,5,10,20}"],["Metrics","Prequential, adaptation, final accuracy, forgetting/BWT"]]},
    results:[{kind:"achieved",title:"Trade-off direction is quantified",metric:"−3.314 pp forgetting",text:"From T=5 to T=20, measured forgetting falls while adaptation and prequential accuracy also decline; this is a resolved observation, not a continual-learning win."},{kind:"non-achieved",title:"Catastrophic forgetting remains",metric:"18.329%",text:"PC T=5 final average accuracy; BP is 18.541%."},{kind:"non-achieved",title:"Stability without plasticity loss",metric:"Rejected",text:"No PC depth satisfies the favorable joint signature."},{kind:"boundary",title:"Seed-123 collapse at T=10",metric:"−6.717 pp",text:"One paired final difference exposes instability hidden by the mean."}],
    table:{headers:["Method","Final","Prequential","Adaptation","Forgetting"],rows:[["BP","18.541%","60.144%","94.561%","96.188%"],["PC T=5","18.329%","57.512%","93.949%","95.687%"],["PC T=10","16.500%","56.487%","91.160%","94.487%"],["PC T=20","16.860%","54.547%","89.829%","92.373%"]]},
    charts:[SPLIT_CURVE],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"More relaxation may reduce interference while preserving new-task acquisition."},{type:"verified",title:"Observation",text:"Forgetting falls from T=5 to T=20, but adaptation and prequential accuracy also fall."},{type:"rejected",title:"Verdict",text:"The favorable stability-without-plasticity-loss hypothesis is rejected."},{type:"interpretation",title:"Interpretation",text:"There is less acquired performance available to forget; low forgetting alone is not success."}],
    sources:[["Evaluation","../../agents/research/pc-relaxation-plasticity/04-evaluation.md"],["Synthesis","../../agents/research/pc-relaxation-plasticity/05-synthesis.md"],["Raw run","../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json"]]
  },

  "width-intervention": {
    title:"Does greater width unlock capacity-dependent gains?", short:"Width intervention", status:"Partial support", statusClass:"warn", source:"Scale study · H1.E2", proposed:false,
    summary:"Double hidden width while freezing the learning recipe and all evaluation regimes.",
    answer:"Width provides narrow method–scenario gains, not a general scale advantage or default replacement.",
    model:MNIST_WIDE,
    experiment:{title:"Width-only intervention",note:"The valid v2 run restores the frozen initialization mapping.",settings:[["Architectures","256×2 control versus 512×2"],["Scenarios","Static, SplitMNIST, Permuted MNIST"],["Seeds","7, 42, 123"],["PC setting","T=5"],["Registered gain gate","≥0.5 pp without >1 pp plasticity loss"],["Invalid artifact","v1 preserved and excluded after RNG-order mismatch"]]},
    results:[{kind:"achieved",title:"Wide PC improves SplitMNIST acquisition",metric:"+7.914 pp",text:"Prequential gain, with +0.528 pp final and +2.575 pp adaptation."},{kind:"achieved",title:"Wide BP improves Permuted mean",metric:"+0.990 pp",text:"One paired seed is negative, so the result remains descriptive."},{kind:"non-achieved",title:"No general width benefit",metric:"Rejected",text:"Static gains miss the registered gate and wide PC does not improve Permuted final accuracy."},{kind:"boundary",title:"Capacity cost",metric:"2.487× params",text:"Width adds parameters and runtime-specific peak memory."}],
    table:{headers:["Architecture","Static BP / PC","Split BP / PC","Permuted BP / PC"],rows:[["256×2","98.267 / 98.193","18.541 / 18.329","91.544 / 91.326"],["512×2","98.490 / 98.447","18.917 / 18.857","92.534 / 91.257"]]},
    charts:[{type:"line",title:"Static validation trajectories under width",subtitle:"Mean across seeds after the shared initialization; width changes only the hidden dimensionality.",x:[1,2,3,4,5,6,7,8,9,10],xLabel:"epoch",yLabel:"validation accuracy (%)",yMin:90,yMax:99,series:[{name:"BP 256×2",color:COLORS.blue,y:[94.742,96.314,97.008,97.356,97.608,97.842,97.961,97.958,98.025,98.122]},{name:"PC 256×2",color:COLORS.green,y:[94.764,96.336,97.019,97.364,97.622,97.800,97.961,97.936,98.042,98.108]},{name:"BP 512×2",color:COLORS.purple,y:[95.928,97.181,97.708,97.917,98.047,98.136,98.236,98.225,98.278,98.308]},{name:"PC 512×2",color:COLORS.orange,y:[95.914,97.136,97.667,97.947,98.044,98.181,98.283,98.239,98.303,98.347]}],takeaway:"Width accelerates the early static curve slightly, but the registered final-accuracy gate is not met."},{type:"bar",title:"Final accuracy across evaluation regimes",subtitle:"The scenario-specific pattern is the central result.",x:["Static","Split","Permuted"],xLabel:"scenario",yLabel:"final accuracy (%)",yMin:0,yMax:100,series:[{name:"BP 256×2",color:COLORS.blue,y:[98.267,18.541,91.544]},{name:"PC 256×2",color:COLORS.green,y:[98.193,18.329,91.326]},{name:"BP 512×2",color:COLORS.purple,y:[98.490,18.917,92.534]},{name:"PC 512×2",color:COLORS.orange,y:[98.447,18.857,91.257]}],takeaway:"Capacity helps selected cells; it does not move every method and scenario together."}],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"Greater width may improve capacity-limited accuracy or plasticity."},{type:"verified",title:"Supported cells",text:"Wide PC improves SplitMNIST acquisition; wide BP improves Permuted mean accuracy."},{type:"rejected",title:"Rejected generalization",text:"Width is not a universal benefit across method and scenario."},{type:"interpretation",title:"Interpretation",text:"Use 512×2 only for targeted follow-up rather than replacing the shared default."}],
    sources:[["Evaluation","../../agents/research/pc-network-scale/04-evaluation.md"],["Synthesis","../../agents/research/pc-network-scale/05-synthesis.md"],["Raw run","../../playground/step1-static-mnist/runs/architecture-sweep-v2.json"]]
  },

  "depth-intervention": {
    title:"Does greater depth expose a PC propagation limit?", short:"Depth intervention", status:"Rejected", statusClass:"fail", source:"Scale study · H2.E1", proposed:false,
    summary:"A preserved negative result: the frozen recipe does not scale cleanly to four hidden layers.",
    answer:"Depth harms both methods, with an additional severe deep-PC continual collapse; the cause remains unresolved.",
    model:MNIST_DEEP,
    experiment:{title:"Depth-only intervention",note:"No post-hoc tuning is allowed in the completed artifact.",settings:[["Architectures","256×2 control versus 256×4"],["Scenarios","Static, SplitMNIST, Permuted MNIST"],["Seeds","7, 42, 123"],["PC setting","T=5 for both depths"],["Optimizer/training","Frozen across depth"],["Validity","Corrected v2 run; invalid v1 preserved and excluded"]]},
    results:[{kind:"achieved",title:"Depth effect is resolved under the frozen recipe",metric:"6 / 6 cells lower",text:"Every method–scenario mean declines at 256×4, providing a consistent negative answer to the registered intervention."},{kind:"non-achieved",title:"Deep PC collapses on Permuted MNIST",metric:"−39.941 pp",text:"Loss versus shallow PC is large and consistent across all seeds."},{kind:"boundary",title:"BP also degrades",metric:"−7.922 pp",text:"The failure is not solely a PC implementation effect."},{kind:"boundary",title:"Inference margin is unresolved",metric:"T=5",text:"Four-layer PC used only the minimum propagation depth."}],
    table:{headers:["Architecture","Static BP / PC","Split BP / PC","Permuted BP / PC"],rows:[["256×2","98.267 / 98.193","18.541 / 18.329","91.544 / 91.326"],["256×4","98.107 / 97.533","12.173 / 11.750","83.622 / 51.385"]]},
    charts:[{type:"line",title:"Static validation trajectories under depth",subtitle:"Mean across seeds after the shared initialization; deep PC learns more slowly even before the continual collapse.",x:[1,2,3,4,5,6,7,8,9,10],xLabel:"epoch",yLabel:"validation accuracy (%)",yMin:88,yMax:99,series:[{name:"BP 256×2",color:COLORS.blue,y:[94.742,96.314,97.008,97.356,97.608,97.842,97.961,97.958,98.025,98.122]},{name:"PC 256×2",color:COLORS.green,y:[94.764,96.336,97.019,97.364,97.622,97.800,97.961,97.936,98.042,98.108]},{name:"BP 256×4",color:COLORS.purple,y:[94.925,96.675,97.264,97.522,97.681,97.769,97.758,97.797,97.794,97.689]},{name:"PC 256×4",color:COLORS.red,y:[91.408,94.436,95.617,95.997,96.572,96.883,97.017,97.119,97.289,97.256]}],takeaway:"Static deep PC recovers partially, but remains slower and ends below the shallow model."},{type:"bar",title:"Depth failure across evaluation regimes",subtitle:"Final mean accuracy under the frozen learning recipe.",x:["Static","Split","Permuted"],xLabel:"scenario",yLabel:"final accuracy (%)",yMin:0,yMax:100,series:[{name:"BP 256×2",color:COLORS.blue,y:[98.267,18.541,91.544]},{name:"PC 256×2",color:COLORS.green,y:[98.193,18.329,91.326]},{name:"BP 256×4",color:COLORS.purple,y:[98.107,12.173,83.622]},{name:"PC 256×4",color:COLORS.red,y:[97.533,11.750,51.385]}],takeaway:"The severe deep-PC Permuted failure is the largest negative result in the scale study."}],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"Greater depth may change PC credit propagation and stability under the same recipe."},{type:"verified",title:"Verified negative result",text:"Depth lowers all six method-scenario means; deep PC fails especially strongly on Permuted MNIST."},{type:"rejected",title:"Verdict",text:"The deep model is rejected under this protocol."},{type:"open",title:"Open mechanism",text:"Depth × relaxation and depth-appropriate optimization must be tested before assigning cause."}],
    sources:[["Evaluation","../../agents/research/pc-network-scale/04-evaluation.md"],["Synthesis","../../agents/research/pc-network-scale/05-synthesis.md"],["Raw run","../../playground/step1-static-mnist/runs/architecture-sweep-v2.json"]]
  },

  "split-cifar-behavior": {
    title:"Stress the matched system on replay-free SplitCIFAR-10", short:"SplitCIFAR behavior", status:"No continual win", statusClass:"fail", source:"CIFAR study · H2.E2", proposed:false,
    summary:"A harder project extension, explicitly separate from the paper's static CIFAR claim.",
    answer:"PC forgets less only while acquiring less; both methods end near catastrophic-forgetting levels.",
    model:CIFAR_MODEL,
    experiment:{title:"Replay-free SplitCIFAR-10",note:"Five class-pair tasks share one ten-way head.",settings:[["Regime","Class incremental"],["Tasks","Five CIFAR-10 class pairs"],["Training","One pass per task; no replay; no model task ID"],["Seeds","7, 42, 123"],["Comparators","Paper-protocol RBP and PC"],["Metrics","Prequential, adaptation, final average, forgetting/BWT"]]},
    results:[{kind:"achieved",title:"Acquisition–retention trade-off is quantified",metric:"−4.533 pp forgetting",text:"PC has lower measured forgetting, but the same runs have lower acquisition, prequential and final accuracy."},{kind:"non-achieved",title:"Both methods catastrophically forget",metric:"17–18%",text:"Final average accuracy after all five tasks."},{kind:"non-achieved",title:"Continual-learning advantage",metric:"Not achieved",text:"The favorable joint stability-plasticity signature is absent."},{kind:"boundary",title:"Separate from Figure 4i",metric:"Project extension",text:"Static reproduction cannot be used as continual-learning evidence."}],
    table:{headers:["Metric","RBP","PC","PC − RBP"],rows:[["Final average","17.990%","16.977%","−1.013 pp"],["Prequential","82.074%","76.042%","−6.033 pp"],["Adaptation","85.863%","81.223%","−4.640 pp"],["Forgetting","85.954%","81.421%","−4.533 pp"]]},
    charts:[SPLIT_CIFAR_PREQUENTIAL,SPLIT_CIFAR_RETENTION,{type:"line",title:"PC inference objective on first task batches",subtitle:"Normalized mean objective across 15 seed×task traces.",x:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],xLabel:"relaxation iteration",yLabel:"objective (% initial)",yMin:35,yMax:105,series:[{name:"PC objective",color:COLORS.purple,y:[100,84.196,79.317,79.039,94.896,71.702,59.644,56.137,53.074,50.608,48.739,47.190,45.748,44.360,43.023,41.741]}],takeaway:"Relaxation is effective overall but not monotonic; iteration count alone is not a convergence certificate."}],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"PC and RBP may differ in acquisition-retention behavior on a causal SplitCIFAR stream."},{type:"verified",title:"Observation",text:"PC forgets less but is worse on adaptation, prequential and final accuracy."},{type:"rejected",title:"Verdict",text:"No continual-learning win is supported; both methods catastrophically forget."},{type:"interpretation",title:"Interpretation",text:"The behavior is a stability-plasticity trade-off, not isolated retention improvement."}],
    sources:[["Evaluation","../../agents/research/predictive-coding-cifar-baseline/04-evaluation.md"],["Synthesis","../../agents/research/predictive-coding-cifar-baseline/05-synthesis.md"],["Raw run","../../playground/step2-cifar10/runs/split-cifar10-v1.json"]]
  },

  "split-cifar-representations": {
    title:"Measure representation drift and label-aligned acquisition", short:"Representation diagnostics", status:"Mechanism not established", statusClass:"warn", source:"CIFAR study · H2.E3", proposed:false,
    summary:"Test whether lower behavioral forgetting corresponds to stable and useful internal representations.",
    answer:"PC changes old-task geometry less at every layer, but also acquires less label-aligned structure; the result is observational, not causal.",
    model:CIFAR_MODEL,
    experiment:{title:"Fixed-anchor representation diagnostics",note:"Instrumentation neutrality is checked against a same-revision diagnostics-off control.",settings:[["Behavioral stream","Same SplitCIFAR-10 protocol as H2.E2"],["Anchors","20 fixed samples per class"],["Layers","conv1, conv2, fc1, logits"],["Checkpoints","Initialization plus five task boundaries"],["Independent units","Three paired seeds; task rows are nested"],["Diagnostics","RDM drift, label alignment, behavior, descriptive correlations"]]},
    results:[{kind:"achieved",title:"PC drift is lower at every layer",metric:"4 / 4",text:"All three seed-level layer means share the same direction."},{kind:"boundary",title:"Early PC geometry is almost frozen",metric:"conv1≈0",text:"Near-zero drift is not automatically useful retention."},{kind:"non-achieved",title:"Causal drift mechanism",metric:"Not established",text:"Correlations are task-confounded and do not identify mediation."},{kind:"boundary",title:"Label-aligned acquisition is also lower",metric:"4 / 4 layers",text:"Reduced drift co-occurs with weaker representational plasticity."}],
    table:{headers:["Layer","RBP drift","PC drift","PC − RBP"],rows:[["conv1","0.009335","0.00000035","−0.009335"],["conv2","0.025365","0.000483","−0.024882"],["fc1","0.362578","0.296496","−0.066082"],["logits","0.870759","0.788836","−0.081923"]]},
    charts:[{type:"line",title:"Representation drift by layer",subtitle:"log₁₀ RDM drift at the final task boundary; more negative means less geometric change.",x:["conv1","conv2","fc1","logits"],xLabel:"layer",yLabel:"log₁₀ RDM drift",yMin:-7,yMax:0,series:[{name:"RBP",color:COLORS.blue,y:[-2.030,-1.596,-0.441,-0.060]},{name:"PC",color:COLORS.orange,y:[-6.456,-3.316,-0.528,-0.103]}],takeaway:"PC is more stable at every measured layer, with almost-frozen early convolutional geometry."},{type:"bar",title:"Initial-to-final label-alignment gain",subtitle:"Representation stability is read jointly with acquisition of label-aligned structure.",x:["conv1","conv2","fc1","logits"],xLabel:"layer",yLabel:"alignment gain",yMin:0,yMax:.12,series:[{name:"RBP",color:COLORS.blue,y:[.021890,.054360,.104883,.074273]},{name:"PC",color:COLORS.orange,y:[.000008,.002808,.074531,.063124]}],takeaway:"PC gains less label-aligned structure at every layer, especially in conv1 and conv2."},SPLIT_CIFAR_RETENTION],
    claims:[{type:"hypothesis",title:"Hypothesis",text:"Lower old-task drift may mediate lower forgetting without simply reducing acquisition."},{type:"verified",title:"Verified observation",text:"PC has lower final-boundary RDM drift at all four layers and weaker label-alignment gains at all four layers."},{type:"interpretation",title:"Interpretation",text:"The evidence is consistent with stability through reduced plasticity or under-learning."},{type:"open",title:"Unresolved mechanism",text:"Acquisition-matched interventions are required before calling low drift protective or causal."}],
    sources:[["Evaluation","../../agents/research/predictive-coding-cifar-baseline/04-evaluation.md"],["Representation summary","../../agents/research/predictive-coding-cifar-baseline/artifacts/h2e3-representation-summary.json"],["Raw diagnostic run","../../playground/step2-cifar10/runs/split-cifar10-representations-v1.json"]]
  }
};

function proposedExperiment(config) {
  return {
    ...config, proposed:true, status:"Proposed · not run", statusClass:"info",
    results:[{kind:"boundary",title:"No completed evidence",metric:"Not run",text:"This page documents a proposed test. No result is included in backed project claims."}],
    table:null, charts:[],
    claims:[
      {type:"hypothesis",title:"Working hypothesis",text:config.workingHypothesis},
      {type:"open",title:"Falsifying outcome",text:config.falsifier},
      {type:"observation",title:"Current evidence boundary",text:"The design has not been executed; results and learning curves are intentionally absent."}
    ]
  };
}

EXPERIMENTS["fair-budget-comparator"] = proposedExperiment({
  title:"Separate conventional BP, repeated-update RBP and PC under fair budgets",short:"Fair-budget comparator",source:"Proposed Q1 refinement",summary:"Resolve the largest comparator ambiguity in the static CIFAR result.",answer:"Planned: equal-sample, equal-update and equal-time panels with validation-only selection.",model:CIFAR_MODEL,
  experiment:{title:"Planned fair-budget comparison",note:"Budget definitions are separate panels rather than one synthetic fairness score.",settings:[["Methods","One-update BP, 16-update RBP, PC-last, PC-all"],["Panels","Equal samples; equal updates; equal wall-clock"],["Selection","Validation only"],["Seeds","To be registered before launch"],["Primary outcome","Accuracy under each explicit budget"],["Status","Not run"]]},
  workingHypothesis:"The apparent method ordering may change when exposure, update count, time and selection information are matched separately.",falsifier:"The ordering and effect size remain stable across all registered fairness panels.",
  sources:[["Canonical proposal","../../research-project-proposal.md"],["Current CIFAR evaluation","../../agents/research/predictive-coding-cifar-baseline/04-evaluation.md"]]
});

EXPERIMENTS["update-alignment-snr"] = proposedExperiment({
  title:"Measure PC update alignment and signal-to-noise",short:"Update alignment & SNR",source:"Proposed W2D3 diagnostic",summary:"Measure update fidelity instead of inferring it from endpoint accuracy.",answer:"Planned: layerwise PC–BP update cosine, norms and a preregistered scale-invariant SNR.",model:MNIST_DEEP,
  experiment:{title:"Planned update diagnostic",note:"Measurements use matched states and batches.",settings:[["Architectures","Two-layer and four-layer MLP"],["Relaxation","Selected T values around the propagation boundary"],["Measures","Update cosine, norm, scale-invariant SNR"],["Sampling unit","Must be fixed before launch"],["Comparator","BP gradient on the matched state/batch"],["Status","Not run"]]},
  workingHypothesis:"The T=1→5 transition and deep-PC failure will appear in layerwise update alignment or SNR.",falsifier:"Update diagnostics remain comparable across successful and failed conditions.",
  sources:[["Canonical proposal","../../research-project-proposal.md"],["Scale synthesis","../../agents/research/pc-network-scale/05-synthesis.md"]]
});

EXPERIMENTS["acquisition-matched-causality"] = proposedExperiment({
  title:"Match new-task acquisition before comparing drift and forgetting",short:"Acquisition-matched causality",source:"Proposed Q2 refinement",summary:"Turn the under-learning alternative explanation into a controlled intervention.",answer:"Planned: match adaptation within ±1 pp, then compare drift, label alignment and forgetting.",model:CIFAR_MODEL,
  experiment:{title:"Planned acquisition-matched intervention",note:"Plasticity is manipulated before interpreting stability.",settings:[["Stream","SplitCIFAR-10"],["Intervention","Early-layer learning rate or plasticity"],["Matching gate","Post-task adaptation within ±1 pp"],["Outcomes","RDM drift, label alignment, forgetting"],["Comparators","Matched PC and RBP conditions"],["Status","Not run"]]},
  workingHypothesis:"If low PC drift is independently protective, it should predict retention after acquisition is matched.",falsifier:"The drift advantage disappears or does not predict retention after acquisition matching.",
  sources:[["Canonical proposal","../../research-project-proposal.md"],["Representation evaluation","../../agents/research/predictive-coding-cifar-baseline/04-evaluation.md"]]
});

EXPERIMENTS["depth-relaxation"] = proposedExperiment({
  title:"Test whether inference budget must grow with network depth",short:"Depth × relaxation",source:"Proposed Q3 refinement",summary:"Resolve one plausible cause of the deep-PC continual collapse.",answer:"Planned: cross two depths with registered relaxation levels and convergence diagnostics.",model:MNIST_DEEP,
  experiment:{title:"Planned depth × relaxation interaction",note:"The completed depth artifact remains frozen and negative.",settings:[["Two-layer PC","T={3,5,10}"],["Four-layer PC","T={5,7,10}"],["Primary regime","Permuted MNIST"],["Secondary regime","SplitMNIST"],["Diagnostics","Per-layer convergence and update quality"],["Status","Not run"]]},
  workingHypothesis:"Four-layer PC requires more than T=5 to achieve a comparable relaxation margin and useful upstream updates.",falsifier:"Additional relaxation fails to improve deep-PC convergence or continual performance.",
  sources:[["Canonical proposal","../../research-project-proposal.md"],["Scale evaluation","../../agents/research/pc-network-scale/04-evaluation.md"]]
});

const EXPERIMENT_ORDER = [
  "cifar-figure-4i", "static-mnist-relaxation", "permuted-mnist-relaxation", "split-mnist-relaxation",
  "width-intervention", "depth-intervention", "split-cifar-behavior", "split-cifar-representations",
  "fair-budget-comparator", "update-alignment-snr", "acquisition-matched-causality", "depth-relaxation"
];
