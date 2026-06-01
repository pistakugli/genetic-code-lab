# 6D ND Recovered-Adjacency Geometry Baseline Court

Generated: 2026-06-01T13:41:21.234Z

## Claim Scope

- Proves if pass: the exact 12-teacher 6D robustness manifest has an executable comparison against several non-GCL baseline families with query budgets, winners, and failure-to-be-SOTA evidence recorded
- Does not claim: SOTA if any measured black-box baseline beats GCL on MAE; topology-perfect ND cell-complex recovery; exact layer attribution; GELU/SwiGLU recovery; arbitrary production API extraction

## Summary

- Pass: true
- Teachers: 12
- Clean/noisy teachers: 4/8
- Profiles: clean, cancellation, near-parallel, low-amplitude, weak-support
- Baseline families: global-linear, global-polynomial, random-feature-relu, local-affine, tree-ensemble
- GCL MAE wins: 12/12
- Best-baseline MAE wins: 0/12
- GCL beats linear ridge: 12/12
- GCL beats quadratic ridge: 12/12
- GCL beats random ReLU ridge: 12/12
- Min GCL region count: 56
- Min GCL adjacency edges: 468
- Max GCL MAE: 0.09301806108793807
- Min GCL R2: 0.9601449089658317
- Median GCL MAE ratio vs best baseline: 0.7684385065146409
- Max GCL MAE ratio vs best baseline: 0.8778374199232243
- Max GCL raw oracle calls: 1501500
- Max baseline direct raw oracle calls: 18000

## Results

### clean-6d-29

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.02514738606591309
- GCL R2: 0.996313022034061
- GCL region count: 65
- GCL adjacency edges: 667
- GCL raw oracle calls: 182000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.07661978601932518
- Best baseline R2: 0.9906587398114751
- GCL MAE ratio vs best baseline: 0.3282100795683556

Ranking:
- gcl-recovered-adjacency: MAE 0.02514738606591309, R2 0.996313022034061, queries 182000
- local-affine-knn-96: MAE 0.07661978601932518, R2 0.9906587398114751, queries 9000
- random-relu-ridge-128: MAE 0.11673668080237043, R2 0.9793320050724412, queries 11375
- quadratic-ridge: MAE 0.16082782656745137, R2 0.9611475250793535, queries 11375
- extra-trees-24x9: MAE 0.2576549938867859, R2 0.8958255256690695, queries 11375
- linear-ridge: MAE 0.3191605501946337, R2 0.8506449757679888, queries 11375

### clean-6d-83

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.03946414407557655
- GCL R2: 0.9956624586754947
- GCL region count: 59
- GCL adjacency edges: 596
- GCL raw oracle calls: 182000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.11070360278467173
- Best baseline R2: 0.9923278011476343
- GCL MAE ratio vs best baseline: 0.3564847311459031

Ranking:
- gcl-recovered-adjacency: MAE 0.03946414407557655, R2 0.9956624586754947, queries 182000
- local-affine-knn-96: MAE 0.11070360278467173, R2 0.9923278011476343, queries 9000
- random-relu-ridge-128: MAE 0.1804527959868152, R2 0.9813923983673911, queries 11375
- quadratic-ridge: MAE 0.23300283952458278, R2 0.965574525606957, queries 11375
- extra-trees-24x9: MAE 0.3907527539514094, R2 0.9109532645984673, queries 11375
- linear-ridge: MAE 0.5242412988079692, R2 0.8345411082491987, queries 11375

### cancellation-6d-717

- Profile: cancellation
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.09301806108793807
- GCL R2: 0.9925411574680458
- GCL region count: 102
- GCL adjacency edges: 1048
- GCL raw oracle calls: 1274000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.12104815193324234
- Best baseline R2: 0.9888933367458964
- GCL MAE ratio vs best baseline: 0.7684385065146409

Ranking:
- gcl-recovered-adjacency: MAE 0.09301806108793807, R2 0.9925411574680458, queries 1274000
- local-affine-knn-96: MAE 0.12104815193324234, R2 0.9888933367458964, queries 9000
- random-relu-ridge-128: MAE 0.18707628358945574, R2 0.9778865091539825, queries 18000
- quadratic-ridge: MAE 0.2578336684024766, R2 0.958663270489265, queries 18000
- extra-trees-24x9: MAE 0.3982716624813128, R2 0.8994541882164802, queries 12000
- linear-ridge: MAE 0.4478026705716438, R2 0.8772874511225205, queries 18000

### near-parallel-6d-805

- Profile: near-parallel
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.014981339622159851
- GCL R2: 0.9601449089658317
- GCL region count: 80
- GCL adjacency edges: 468
- GCL raw oracle calls: 1501500
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.051357273787714985
- Best baseline R2: 0.8760272103087766
- GCL MAE ratio vs best baseline: 0.291708233659075

Ranking:
- gcl-recovered-adjacency: MAE 0.014981339622159851, R2 0.9601449089658317, queries 1501500
- local-affine-knn-96: MAE 0.051357273787714985, R2 0.8760272103087766, queries 9000
- random-relu-ridge-128: MAE 0.08857208105828804, R2 0.7283407399687587, queries 18000
- extra-trees-24x9: MAE 0.10418166761264609, R2 0.5974568924439174, queries 12000
- quadratic-ridge: MAE 0.12094112505891756, R2 0.5212071268407694, queries 18000
- linear-ridge: MAE 0.12689920179199365, R2 0.4375738223274773, queries 18000

### low-amplitude-6d-634

- Profile: low-amplitude
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.031089536651472263
- GCL R2: 0.992196673628252
- GCL region count: 77
- GCL adjacency edges: 527
- GCL raw oracle calls: 910000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.037227880350513186
- Best baseline R2: 0.9889333486409498
- GCL MAE ratio vs best baseline: 0.8351143379304348

Ranking:
- gcl-recovered-adjacency: MAE 0.031089536651472263, R2 0.992196673628252, queries 910000
- local-affine-knn-96: MAE 0.037227880350513186, R2 0.9889333486409498, queries 9000
- random-relu-ridge-128: MAE 0.05786408159883345, R2 0.9760048485302184, queries 18000
- quadratic-ridge: MAE 0.0951805510382065, R2 0.934002942129862, queries 18000
- extra-trees-24x9: MAE 0.12710123510611743, R2 0.8867868620785937, queries 12000
- linear-ridge: MAE 0.16054218841170873, R2 0.7964993541873276, queries 18000

### weak-support-6d-911

- Profile: weak-support
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.04180015414938247
- GCL R2: 0.9871745801121442
- GCL region count: 58
- GCL adjacency edges: 520
- GCL raw oracle calls: 1001000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.04948786046568404
- Best baseline R2: 0.9834609259036211
- GCL MAE ratio vs best baseline: 0.8446547043262783

Ranking:
- gcl-recovered-adjacency: MAE 0.04180015414938247, R2 0.9871745801121442, queries 1001000
- local-affine-knn-96: MAE 0.04948786046568404, R2 0.9834609259036211, queries 9000
- random-relu-ridge-128: MAE 0.08932258978916617, R2 0.9521209307200792, queries 18000
- quadratic-ridge: MAE 0.1021345419577603, R2 0.9341787148525629, queries 18000
- extra-trees-24x9: MAE 0.15255978439427104, R2 0.847388472265666, queries 12000
- linear-ridge: MAE 0.2911996697115297, R2 0.4699150451460802, queries 18000

### clean-6d-151

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.02706279487397642
- GCL R2: 0.9970428214183646
- GCL region count: 56
- GCL adjacency edges: 469
- GCL raw oracle calls: 234000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.12162154249024444
- Best baseline R2: 0.9886792535465146
- GCL MAE ratio vs best baseline: 0.2225164581854172

Ranking:
- gcl-recovered-adjacency: MAE 0.02706279487397642, R2 0.9970428214183646, queries 234000
- local-affine-knn-96: MAE 0.12162154249024444, R2 0.9886792535465146, queries 9000
- random-relu-ridge-128: MAE 0.20064787795671607, R2 0.9757091737971441, queries 14625
- quadratic-ridge: MAE 0.2645911132715835, R2 0.9564514510946953, queries 14625
- extra-trees-24x9: MAE 0.36011944798872797, R2 0.8992424486625701, queries 12000
- linear-ridge: MAE 0.7267849164355925, R2 0.6844780109586222, queries 14625

### clean-6d-211

- Profile: clean
- Noisy oracle: false
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.037607319823290956
- GCL R2: 0.9971755828213908
- GCL region count: 57
- GCL adjacency edges: 619
- GCL raw oracle calls: 182000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.12607379597732338
- Best baseline R2: 0.9916310635691375
- GCL MAE ratio vs best baseline: 0.2982960854930973

Ranking:
- gcl-recovered-adjacency: MAE 0.037607319823290956, R2 0.9971755828213908, queries 182000
- local-affine-knn-96: MAE 0.12607379597732338, R2 0.9916310635691375, queries 9000
- random-relu-ridge-128: MAE 0.2261376440854124, R2 0.9762200737416434, queries 11375
- quadratic-ridge: MAE 0.2603744231601138, R2 0.9696667749533494, queries 11375
- extra-trees-24x9: MAE 0.36364039134436105, R2 0.9332338503606612, queries 11375
- linear-ridge: MAE 0.6391590844880918, R2 0.8160226504354596, queries 11375

### cancellation-6d-742

- Profile: cancellation
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.05805537550049842
- GCL R2: 0.9926343502227222
- GCL region count: 91
- GCL adjacency edges: 960
- GCL raw oracle calls: 1092000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.07349018815460347
- Best baseline R2: 0.9896984592922534
- GCL MAE ratio vs best baseline: 0.7899745116772

Ranking:
- gcl-recovered-adjacency: MAE 0.05805537550049842, R2 0.9926343502227222, queries 1092000
- local-affine-knn-96: MAE 0.07349018815460347, R2 0.9896984592922534, queries 9000
- random-relu-ridge-128: MAE 0.1215248423248104, R2 0.9738838316099667, queries 18000
- quadratic-ridge: MAE 0.1231910474622956, R2 0.9733085474436268, queries 18000
- extra-trees-24x9: MAE 0.2288120730902971, R2 0.9055411837189284, queries 12000
- linear-ridge: MAE 0.3307664632443247, R2 0.8157192045825366, queries 18000

### near-parallel-6d-829

- Profile: near-parallel
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.018791016075038764
- GCL R2: 0.9785253611754295
- GCL region count: 95
- GCL adjacency edges: 729
- GCL raw oracle calls: 1183000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.037826279698820235
- Best baseline R2: 0.9749266840867121
- GCL MAE ratio vs best baseline: 0.4967714568986502

Ranking:
- gcl-recovered-adjacency: MAE 0.018791016075038764, R2 0.9785253611754295, queries 1183000
- local-affine-knn-96: MAE 0.037826279698820235, R2 0.9749266840867121, queries 9000
- random-relu-ridge-128: MAE 0.0670818948169614, R2 0.9403964965231257, queries 18000
- extra-trees-24x9: MAE 0.0733188176064956, R2 0.9114202304838682, queries 12000
- quadratic-ridge: MAE 0.07717213971240097, R2 0.9294236672089703, queries 18000
- linear-ridge: MAE 0.20024319487180386, R2 0.5295644592241024, queries 18000

### low-amplitude-6d-658

- Profile: low-amplitude
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.0342815325985024
- GCL R2: 0.9941575530659655
- GCL region count: 65
- GCL adjacency edges: 538
- GCL raw oracle calls: 910000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.03905225708138605
- Best baseline R2: 0.9926148093276596
- GCL MAE ratio vs best baseline: 0.8778374199232243

Ranking:
- gcl-recovered-adjacency: MAE 0.0342815325985024, R2 0.9941575530659655, queries 910000
- local-affine-knn-96: MAE 0.03905225708138605, R2 0.9926148093276596, queries 9000
- random-relu-ridge-128: MAE 0.05385879306956264, R2 0.986918650334717, queries 18000
- quadratic-ridge: MAE 0.06958988344462302, R2 0.9783132322533177, queries 18000
- linear-ridge: MAE 0.0941041913491464, R2 0.961322212085739, queries 18000
- extra-trees-24x9: MAE 0.12862304650411385, R2 0.929397147305333, queries 12000

### weak-support-6d-937

- Profile: weak-support
- Noisy oracle: true
- Winner: gcl-recovered-adjacency
- GCL MAE: 0.08742366261199823
- GCL R2: 0.9943721976433024
- GCL region count: 112
- GCL adjacency edges: 1069
- GCL raw oracle calls: 1001000
- Best baseline: local-affine-knn-96
- Best baseline MAE: 0.10045298312308844
- Best baseline R2: 0.9930493510634893
- GCL MAE ratio vs best baseline: 0.8702943396402182

Ranking:
- gcl-recovered-adjacency: MAE 0.08742366261199823, R2 0.9943721976433024, queries 1001000
- local-affine-knn-96: MAE 0.10045298312308844, R2 0.9930493510634893, queries 9000
- random-relu-ridge-128: MAE 0.1597808571972155, R2 0.9828760919651205, queries 18000
- quadratic-ridge: MAE 0.18332568119109866, R2 0.9764921604168721, queries 18000
- extra-trees-24x9: MAE 0.3154331376737245, R2 0.9300792055970728, queries 12000
- linear-ridge: MAE 0.3933276192757912, R2 0.8950443939207683, queries 18000

## Conclusion

GCL won MAE against every measured baseline on this 6D suite; this court is release-gated, and the next step is stronger package-backed external baselines plus adaptive-adversarial teacher generation.
