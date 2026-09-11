import numpy as np
from src.homography import estimate_homography_dlt,ransac_homography,transform_points
def test_identity_and_translation():
 p=np.array([[0,0],[1,0],[1,1],[0,1],[2,3]],float); assert np.allclose(transform_points(p,np.eye(3)),p); H=np.array([[1,0,50],[0,1,20],[0,0,1.]])
 got=estimate_homography_dlt(p,transform_points(p,H));assert np.allclose(got,H,atol=1e-8)
def test_projective_ransac_with_30_percent_outliers():
 rng=np.random.default_rng(1);p=rng.uniform(0,200,(100,2));H=np.array([[1.02,.03,12],[-.02,.98,8],[1e-4,-2e-4,1.] ]);q=transform_points(p,H);q[:30]=rng.uniform(0,200,(30,2));r=ransac_homography(p,q,{'reprojection_threshold':1.,'confidence':.995,'max_iterations':5000,'seed':42});assert r.inliers.sum()>=69;assert np.allclose(r.H/H[2,2],H,atol=1e-5)
