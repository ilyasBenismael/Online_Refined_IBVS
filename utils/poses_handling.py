from utils.lin_algeb import LinAlgeb






class PosesHandling :


    def get_sfm_poses(recon) : 

        homog_poses = {}
        for image in recon.images.values():
            if not image.has_pose:
                continue
            T_cw = image.cam_from_world()   # Rigid3d
            R = T_cw.rotation.matrix()      # (3,3)
            t = T_cw.translation            # (3,)
            T_h = LinAlgeb.get_homog_frm_rt(R,t)
            homog_poses[image.name] = T_h

        return homog_poses


