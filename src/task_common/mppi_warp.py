"""Warp kernels of the MPPI solver: the fused LCS rollout and the clearance distances.

Same maths as the torch paths (``TorchLcs.rollout``, ``Clearance.torch_path``) in float32, one
thread per sample, launched on torch's current stream so the solve stays CUDA-graph capturable.
``warp`` is imported lazily (CPU-only users never load it).
"""

import numpy as np
import torch

from task_common.mppi_costs import N_P


def _wp_funcs():
    import warp as wp

    @wp.func
    def quat_exp(r: wp.vec3):
        a = wp.length(r)
        s = 0.5 - a * a / 48.0
        if a > 1.0e-4:
            s = wp.sin(0.5 * a) / a
        return wp.quat(s * r[0], s * r[1], s * r[2], wp.cos(0.5 * a))

    @wp.func
    def quat_angle(q: wp.quat):
        return 2.0 * wp.atan2(wp.length(wp.vec3(q[0], q[1], q[2])), wp.abs(q[3]))

    return quat_exp, quat_angle


# --- LCS rollout --------------------------------------------------------------------------------

def _fused_kernel(nx: int, nl: int, nu: int):
    import warp as wp

    _wp_quat_exp, _wp_quat_angle = _wp_funcs()

    f32 = wp.float32
    vx, vl = wp.types.vector(nx, f32), wp.types.vector(nl, f32)
    vu, v6 = wp.types.vector(nu, f32), wp.types.vector(N_P, f32)
    mxx, mxu = wp.types.matrix((nx, nx), f32), wp.types.matrix((nx, nu), f32)
    mxl, mx6 = wp.types.matrix((nx, nl), f32), wp.types.matrix((nx, N_P), f32)
    mlx, mlu = wp.types.matrix((nl, nx), f32), wp.types.matrix((nl, nu), f32)
    mll, ml6 = wp.types.matrix((nl, nl), f32), wp.types.matrix((nl, N_P), f32)

    @wp.kernel(module="unique", enable_backward=False)
    def rollout(z0: wp.array(dtype=vx), p0: wp.array(dtype=v6), r0: wp.array(dtype=wp.vec3),
                U: wp.array2d(dtype=vu),
                A: wp.array(dtype=mxx), B: wp.array(dtype=mxu), D: wp.array(dtype=mxl),
                d: wp.array(dtype=vx), Ap: wp.array(dtype=mx6), sE: wp.array(dtype=mlx),
                sH: wp.array(dtype=mlu), sc: wp.array(dtype=vl), sEp: wp.array(dtype=ml6),
                Ms: wp.array(dtype=mll), betas: wp.array(dtype=f32), n_pgd: int, H: int,
                Z: wp.array2d(dtype=vx), P: wp.array2d(dtype=v6), R: wp.array2d(dtype=wp.vec2)):
        k = wp.tid()
        z = z0[0]
        p = p0[0]
        qf = _wp_quat_exp(r0[0])
        qu = _wp_quat_exp(r0[1])
        Z[k, 0] = z
        P[k, 0] = p
        R[k, 0] = wp.vec2(_wp_quat_angle(qf), _wp_quat_angle(qu))
        zero = vl()
        for h in range(H):
            u = U[k, h]
            nsq = -(sE[0] @ z + sH[0] @ u + sc[0] + sEp[0] @ p)
            lam = vl()
            y = vl()
            for i in range(n_pgd):
                lam_old = lam
                # lam = max(y - s (F y + q), 0) with Ms = I - s F
                lam = wp.max(nsq + Ms[0] @ y, zero)
                y = lam_old + (1.0 + betas[i]) * (lam - lam_old)
            z = A[0] @ z + B[0] @ u + D[0] @ lam + d[0] + Ap[0] @ p
            p = p + v6(u[0], u[1], u[2], u[3], u[4], u[5])
            qf = _wp_quat_exp(wp.vec3(u[6], u[7], u[8])) * qf
            qu = _wp_quat_exp(wp.vec3(u[9], u[10], u[11])) * qu
            Z[k, h + 1] = z
            P[k, h + 1] = p
            R[k, h + 1] = wp.vec2(_wp_quat_angle(qf), _wp_quat_angle(qu))

    types = {"vx": vx, "v6": v6, "vu": vu, "mxx": mxx, "mxu": mxu, "mxl": mxl, "mx6": mx6,
             "mlx": mlx, "mlu": mlu, "vl": vl, "ml6": ml6, "mll": mll}
    return rollout, types


class _FusedRollout:
    """``TorchLcs.rollout_fused``: the deploy's matrices as Warp arrays + one launch per call."""

    def __init__(self, lcs) -> None:
        import warp as wp

        self.wp, self.lcs = wp, lcs
        self.kernel, ty = _fused_kernel(lcs.n_x, lcs.n_lam, lcs.n_u)
        self.ty = ty
        dev = str(lcs.device) if lcs.device.index is not None else "cuda:0"
        self.dev = dev
        s = lcs.step_size
        f_pgd = lcs.np["F_pgd"]

        def arr(x, t):
            return wp.array([np.asarray(x, dtype=np.float32)], dtype=t, device=dev)

        n = lcs.np
        self.mats = [arr(n["A"], ty["mxx"]), arr(n["B"], ty["mxu"]), arr(n["D"], ty["mxl"]),
                     arr(n["d"], ty["vx"]), arr(n["A_p"], ty["mx6"]), arr(s * n["E"], ty["mlx"]),
                     arr(s * n["H"], ty["mlu"]), arr(s * n["c"], ty["vl"]),
                     arr(s * n["E_p"], ty["ml6"]),
                     arr(np.eye(lcs.n_lam) - s * f_pgd, ty["mll"])]
        self._betas: dict[int, object] = {}

    def __call__(self, z0, p0, U, rot0, iters):
        wp, lcs = self.wp, self.lcs
        K, H, _ = U.shape
        if int(iters) not in self._betas:
            self._betas[int(iters)] = wp.array(np.asarray(lcs.nesterov_betas(iters), np.float32),
                                               dtype=wp.float32, device=self.dev)
        z0 = lcs.tensor(z0).reshape(1, lcs.n_x).contiguous()
        p0 = lcs.tensor(p0).reshape(1, N_P).contiguous()
        U = U.contiguous()
        r0 = lcs.tensor(np.zeros((2, 3)) if rot0 is None else rot0).reshape(2, 3).contiguous()
        Z = torch.empty((K, H + 1, lcs.n_x), dtype=torch.float32, device=lcs.device)
        P = torch.empty((K, H + 1, N_P), dtype=torch.float32, device=lcs.device)
        R = torch.empty((K, H + 1, 2), dtype=torch.float32, device=lcs.device)
        ty = self.ty
        wp.launch(self.kernel, dim=K, device=self.dev,
                  stream=wp.stream_from_torch(torch.cuda.current_stream(lcs.device)),
                  inputs=[wp.from_torch(z0, dtype=ty["vx"]), wp.from_torch(p0, dtype=ty["v6"]),
                          wp.from_torch(r0, dtype=wp.vec3), wp.from_torch(U, dtype=ty["vu"]),
                          *self.mats, self._betas[int(iters)], int(iters), int(H)],
                  outputs=[wp.from_torch(Z, dtype=ty["vx"]), wp.from_torch(P, dtype=ty["v6"]),
                           wp.from_torch(R, dtype=wp.vec2)])
        return Z, P, R


# --- clearance (gripper sets vs plate + pulleys) ------------------------------------------------

def _clearance_kernel():
    import warp as wp

    _wp_quat_exp, _ = _wp_funcs()
    f32 = wp.float32
    vu, v6 = wp.types.vector(12, f32), wp.types.vector(N_P, f32)

    @wp.func
    def poly_dist(r: f32, z: f32, prof: wp.array2d(dtype=wp.vec2), j: int):
        # padded vertices give zero-length edges (t = 0)
        n = prof.shape[1]
        d2 = f32(1.0e30)
        inside = wp.int32(0)
        for i in range(n):
            A = prof[j, i]
            B = prof[j, (i + 1) % n]
            abx = B[0] - A[0]
            abz = B[1] - A[1]
            apx = r - A[0]
            apz = z - A[1]
            den = abx * abx + abz * abz
            t = f32(0.0)
            if den > f32(0.0):
                t = wp.clamp((apx * abx + apz * abz) / den, f32(0.0), f32(1.0))
            dx = apx - t * abx
            dz = apz - t * abz
            d2 = wp.min(d2, dx * dx + dz * dz)
            if (A[1] > z and B[1] <= z) or (A[1] <= z and B[1] > z):  # noqa: SIM102
                if r < A[0] + (z - A[1]) / abz * abx:
                    inside = 1 - inside
        d = wp.sqrt(d2)
        if inside == 1:
            d = -d
        return d

    @wp.func
    def box_sd(x: wp.vec3, h: wp.vec3):
        q = wp.vec3(wp.abs(x[0]) - h[0], wp.abs(x[1]) - h[1], wp.abs(x[2]) - h[2])
        out = wp.length(wp.vec3(wp.max(q[0], f32(0.0)), wp.max(q[1], f32(0.0)),
                                wp.max(q[2], f32(0.0))))
        return out + wp.min(wp.max(q[0], wp.max(q[1], q[2])), f32(0.0))

    @wp.func
    def cyl_bound(x: wp.vec3, lim: wp.vec2):
        a = wp.sqrt(x[0] * x[0] + x[1] * x[1]) - lim[0]
        b = wp.abs(x[2]) - lim[1]
        pa = wp.max(a, f32(0.0))
        pb = wp.max(b, f32(0.0))
        return wp.sqrt(pa * pa + pb * pb) + wp.min(wp.max(a, b), f32(0.0))

    @wp.func
    def set_dist(pts: wp.array(dtype=wp.vec3), cl: wp.array(dtype=wp.vec2i),
                 cc: wp.array(dtype=wp.vec4), c0: int, stride: int, Rq: wp.mat33, t: wp.vec3,
                 plate_R: wp.array(dtype=wp.mat33), plate_p: wp.array(dtype=wp.vec3),
                 plate_h: wp.array(dtype=wp.vec3), pul_R: wp.array(dtype=wp.mat33),
                 pul_p: wp.array(dtype=wp.vec3), pul_b: wp.array(dtype=wp.vec2),
                 prof: wp.array2d(dtype=wp.vec2)):
        # Board.min_distance; a cluster is skipped when centre bound - radius >= best.
        RbT = wp.transpose(plate_R[0])
        Mb = RbT * Rq
        bb = RbT * (t - plate_p[0])
        h = plate_h[0]
        best = f32(1.0e30)
        for ci in range(c0, cl.shape[0], stride):
            c = cc[ci]
            if box_sd(Mb * wp.vec3(c[0], c[1], c[2]) + bb, h) - c[3] < best:
                for i in range(cl[ci][0], cl[ci][1]):
                    best = wp.min(best, box_sd(Mb * pts[i] + bb, h))
        for j in range(pul_R.shape[0]):
            RjT = wp.transpose(pul_R[j])
            Mj = RjT * Rq
            bj = RjT * (t - pul_p[j])
            lim = pul_b[j]
            for ci in range(c0, cl.shape[0], stride):
                c = cc[ci]
                if cyl_bound(Mj * wp.vec3(c[0], c[1], c[2]) + bj, lim) - c[3] < best:
                    for i in range(cl[ci][0], cl[ci][1]):
                        x = Mj * pts[i] + bj
                        if cyl_bound(x, lim) < best:
                            best = wp.min(best, poly_dist(wp.sqrt(x[0] * x[0] + x[1] * x[1]),
                                                          x[2], prof, j))
        return best

    @wp.kernel(module="unique", enable_backward=False)
    def clear(P: wp.array2d(dtype=v6), U: wp.array2d(dtype=vu), q0: wp.array2d(dtype=wp.quat),
              pts_ur: wp.array(dtype=wp.vec3), cl_ur: wp.array(dtype=wp.vec2i),
              cc_ur: wp.array(dtype=wp.vec4), pts_f: wp.array(dtype=wp.vec3),
              cl_f: wp.array(dtype=wp.vec2i), cc_f: wp.array(dtype=wp.vec4),
              plate_R: wp.array(dtype=wp.mat33), plate_p: wp.array(dtype=wp.vec3),
              plate_h: wp.array(dtype=wp.vec3), pul_R: wp.array(dtype=wp.mat33),
              pul_p: wp.array(dtype=wp.vec3), pul_b: wp.array(dtype=wp.vec2),
              prof: wp.array2d(dtype=wp.vec2), n_chunks: int, C: wp.array3d(dtype=f32)):
        k, h, a, c = wp.tid()
        arm = 1 - a  # C: UR first; P / U / q0: Franka first
        q = q0[k % q0.shape[0], arm]
        for j in range(h):
            u = U[k, j]
            q = _wp_quat_exp(wp.vec3(u[6 + 3 * arm], u[7 + 3 * arm], u[8 + 3 * arm])) * q
        p = P[k, h]
        t = wp.vec3(p[3 * arm], p[3 * arm + 1], p[3 * arm + 2])
        Rq = wp.quat_to_matrix(q)
        d = f32(0.0)
        if a == 0:
            d = set_dist(pts_ur, cl_ur, cc_ur, c, n_chunks, Rq, t, plate_R, plate_p, plate_h,
                         pul_R, pul_p, pul_b, prof)
        else:
            d = set_dist(pts_f, cl_f, cc_f, c, n_chunks, Rq, t, plate_R, plate_p, plate_h,
                         pul_R, pul_p, pul_b, prof)
        wp.atomic_min(C, k, h, a, d)

    return clear, vu, v6
