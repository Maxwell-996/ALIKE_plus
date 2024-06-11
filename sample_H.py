import torch
import scipy
from scipy.stats import truncnorm
import numpy as np
def sample_homography(shape,
                      perspective=True, 
                      scaling=True, 
                      rotation=True, 
                      translation=True,
                      n_scales=5, 
                      n_angles=25, 
                      scaling_amplitude=0.2, 
                      perspective_amplitude_x=0.2,
                      perspective_amplitude_y=0.2, 
                      patch_ratio=0.85, 
                      max_angle=torch.pi/2,
                      allow_artifacts=True,
                      translation_overflow=0.):

    margin = (1 -patch_ratio) / 2
    pts1 = margin + torch.tensor([[0., 0.], [0., patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], dtype=torch.float32)
    pts2 = pts1.clone()

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        mean = 0
        norm_geney = scipy.stats.truncnorm(a = -2, b = 2, \
                                           loc=mean, scale=perspective_amplitude_y/2)
        norm_genex = scipy.stats.truncnorm(a = -2, b = 2, \
                                           loc=mean, scale=perspective_amplitude_x/2)
        target = norm_genex.rvs(size = 2)
        perspective_displacement = norm_geney.rvs(size = 1)[0]
        h_displacement_left , h_displacement_right = target[0],target[1]
        pts2 += torch.tensor(np.array([
            [h_displacement_left, perspective_displacement],
            [h_displacement_left, -perspective_displacement],
            [h_displacement_right, perspective_displacement],
            [h_displacement_right, -perspective_displacement]
        ]))

    if scaling:
        norm_gene = scipy.stats.truncnorm(a = -2, b = 2, \
                                           loc=1, scale=scaling_amplitude/2)
        scales_ = norm_gene.rvs(size = n_scales)
        scales = torch.tensor(np.concatenate(([1.], scales_)))
        center = torch.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center).unsqueeze(0) * scales.reshape((n_scales + 1,1,1)) + center
        if allow_artifacts:
            valid = torch.arange(1, n_scales + 1)
        else:
            valid = torch.all(((scaled>=0.) & (scaled<1.)).reshape(n_scales + 1,-1),dim = -1)
        #scaled = scaled[valid]
        idx = np.random.randint(low = 0, high = scaled.shape[0])
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = -torch.min(pts2, dim=0).values, torch.min(1 - pts2, dim=0).values
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += torch.tensor(np.array([np.random.rand()*(t_max[0]-t_min[0])+t_min[0],
                                        np.random.rand()*(t_max[1]-t_min[1])+t_min[1]])).unsqueeze(0)

    if rotation:
        angles = torch.tensor(np.linspace(-max_angle, max_angle, n_angles))
        angles = torch.tensor(np.concatenate(([0.], angles))).unsqueeze(0)
        center = torch.mean(pts2, dim=0, keepdims=True)
        rot_mat = torch.concat([torch.cos(angles), -torch.sin(angles), torch.sin(angles),torch.cos(angles)],dim = 0).permute(1,0).reshape(-1,2,2)
        rotated = torch.matmul((pts2 - center).unsqueeze(0).repeat((n_angles+1, 1, 1)), rot_mat) + center
        if allow_artifacts:
            valid = torch.arange(1, n_angles + 1)
        else:
            valid = torch.all(((rotated>=0.) & (rotated<1.)).reshape(n_angles + 1,-1),dim = -1) 
        #rotated = rotated[valid]
        idx =  np.random.randint(low = 0, high = rotated.shape[0])
        pts2 = rotated[idx]

    shape =torch.tensor(shape[::-1], dtype=torch.float32)
    pts1 = pts1*( shape.unsqueeze(0) )
    pts2 = pts2*( shape.unsqueeze(0) )

    def ax(p, q): 
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): 
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = torch.tensor([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)])

    p_mat = torch.tensor([[pts2[i][j] for i in range(4) for j in range(2)]]).permute(1,0)
    x = torch.linalg.solve(a_mat,p_mat)

    homography = x.permute(1,0)
    
    return homography,pts1,pts2

if __name__ == "__main__":
    while(1):
        x = sample_homography((720,960))
        print(x)