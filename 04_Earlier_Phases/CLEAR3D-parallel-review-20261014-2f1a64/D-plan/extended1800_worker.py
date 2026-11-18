import sys
sys.path[:0]=['/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src','/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/scripts']
import lidc3d.diagnostic_training as c
c.RESEARCH_LIMITS={**c.RESEARCH_LIMITS,'total_seconds':1800,'run_seconds':1800}
from diagnostic_training_worker import main
if __name__=='__main__':
    raise SystemExit(main())
