#!/usr/bin/env python3
"""
  N-ary fuzzy CSG, v7: normalization ablations.

Methods:
  A) BASELINE          -  overcomplete discrete pool (12x5=60 prims)
  G) BC-TYPE + SINE    -  Bc product type selection
  K) BC-SIZE           -  Bc + mean-param size normalization
  N) BC-VOL            -  Bc + volume-radius normalization
  O) BC-AREA           -  Bc + surface-area normalization
  P) BC-K+J            -  Bc + size * local grad normalization (stacked)

Usage:
  python nary_benchmark_v7.py --mesh ./bunny.obj --epochs 30000
"""

import os, sys, time, math, json, argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import mcubes
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import igl; HAS_IGL = True
except ImportError:
    HAS_IGL = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

MIN_SIZE = 0.02

# MESH SDF
class MeshSDF:
    def __init__(self, mesh_path, normalize=True, margin=0.1):
        print(f"Loading mesh: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = trimesh.util.concatenate(list(self.mesh.geometry.values()))
        if not self.mesh.is_watertight:
            self.mesh = self._repair(self.mesh)
        if normalize:
            self.mesh.vertices -= self.mesh.bounding_box.centroid
            s = (1.0 - margin) / np.abs(self.mesh.vertices).max()
            self.mesh.vertices *= s
        self.vertices = np.ascontiguousarray(self.mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.mesh.faces, dtype=np.int64)
        print(f"  {len(self.vertices):,} verts, watertight={self.mesh.is_watertight}")
    def _repair(self, mesh):
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces); mf.repair(verbose=False)
            f = mf.mesh.faces; f = f.reshape(-1,4)[:,1:4] if f.ndim==1 else f
            r = trimesh.Trimesh(vertices=mf.mesh.points, faces=f)
            if r.is_watertight: print("  ✓ Repaired"); return r
        except: pass
        try:
            from scipy import ndimage; res=128
            vox=mesh.voxelized(pitch=2.0/res)
            vg=ndimage.binary_fill_holes(vox.matrix).astype(np.float32)
            vg=ndimage.gaussian_filter(vg,sigma=0.3)
            vg=np.pad(vg,1,mode='constant',constant_values=0)
            v,t=mcubes.marching_cubes(vg,0.5)
            v=(v-1)/res*(mesh.vertices.max()-mesh.vertices.min())+mesh.vertices.min()
            return trimesh.Trimesh(vertices=v,faces=t)
        except: pass
        return mesh
    def query(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64)
        if HAS_IGL:
            sdf,_,_,_ = igl.signed_distance(points, self.vertices, self.faces)
            return sdf.astype(np.float32)
        _,dist,_ = self.mesh.nearest.on_surface(points)
        sdf=dist.copy(); sdf[self.mesh.contains(points)] *= -1
        return sdf.astype(np.float32)
    def query_sdf_grid(self, resolution):
        idx=np.linspace(-1,1,resolution)
        x,y,z=np.meshgrid(idx,idx,idx,indexing='ij')
        return self.query(np.stack([x.flatten(),y.flatten(),z.flatten()],1)).reshape(resolution,resolution,resolution)
    def sample_points(self, n, surface_ratio=0.3, near_ratio=0.5, near_std=0.03):
        ns=int(n*surface_ratio); nn_=int(n*near_ratio); nr=n-ns-nn_
        ps,_=trimesh.sample.sample_surface(self.mesh,ns)
        pn,_=trimesh.sample.sample_surface(self.mesh,nn_)
        pn=np.clip(pn+np.random.randn(nn_,3)*near_std,-0.99,0.99)
        pr=np.random.uniform(-1,1,(nr,3))
        pts=np.vstack([ps,pn,pr]).astype(np.float32)
        occ=(self.query(pts)<0).astype(np.float32)
        perm=np.random.permutation(len(pts))
        return pts[perm], occ[perm]

# UTILITIES
def sample_3D_grid(resolution, low=-1.0, high=1.0):
    idx=torch.linspace(low,high,resolution,device=device)
    x,y,z=torch.meshgrid(idx,idx,idx,indexing='ij')
    return torch.stack([x.flatten(),y.flatten(),z.flatten()],dim=1)
def mc_from_occ(fp, occ, res):
    o=occ.detach().cpu().numpy().reshape(res,res,res)
    if o.min()>=0.5 or o.max()<=0.5: return None
    v,t=mcubes.marching_cubes(-(o*2-1),0)
    v=v/(res-1)*2-1;t=t.copy();t[:,[0,1]]=t[:,[1,0]]
    mcubes.export_obj(v,t,fp); return v,t
def mc_from_sdf(fp, sg, res):
    if sg.min()>=0 or sg.max()<=0: return None
    v,t=mcubes.marching_cubes(sg,0);v=v/(res-1)*2-1;t=t.copy();t[:,[0,1]]=t[:,[1,0]]
    mcubes.export_obj(v,t,fp); return v,t
def compute_iou(pred,gt,threshold=0.5):
    p=(pred>threshold).float();g=(gt>threshold).float()
    return ((p*g).sum()/(((p+g)>0).float().sum()+1e-8)).item()
def compute_fscore(pred,gt,threshold=0.5):
    p=(pred>threshold).float();g=(gt>threshold).float();tp=(p*g).sum()
    return (2*tp/(p.sum()+g.sum()+1e-8)).item()
def batched_forward(model, grid, chunk=32**3, **kw):
    with torch.no_grad():
        return torch.cat([model(b,**kw) for b in torch.split(grid,chunk)],0)
def euler_to_R(a):
    cx,sx=torch.cos(a[0]),torch.sin(a[0])
    cy,sy=torch.cos(a[1]),torch.sin(a[1])
    cz,sz=torch.cos(a[2]),torch.sin(a[2])
    R=torch.zeros(3,3,device=a.device,dtype=a.dtype)
    R[0,0]=cy*cz;R[0,1]=cz*sx*sy-cx*sz;R[0,2]=cx*cz*sy+sx*sz
    R[1,0]=cy*sz;R[1,1]=cx*cz+sx*sy*sz;R[1,2]=cx*sy*sz-cz*sx
    R[2,0]=-sy;R[2,1]=cy*sx;R[2,2]=cx*cy
    return R

# SDF PRIMITIVES
def sdf_sphere_local(pl,r):
    return torch.sqrt((pl**2).sum(-1)+1e-8)-(r.abs()+MIN_SIZE)
def sdf_block_local(pl,half_size):
    h=half_size.abs()+MIN_SIZE;d=torch.abs(pl)-h
    return torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
def sdf_cylinder_local(pl,r,hh):
    r_=r.abs()+MIN_SIZE;hh_=hh.abs()+MIN_SIZE
    dr=torch.sqrt(pl[:,0]**2+pl[:,2]**2+1e-8)-r_
    dh=torch.abs(pl[:,1])-hh_
    return torch.sqrt(torch.clamp(dr,min=0)**2+torch.clamp(dh,min=0)**2+1e-8)+torch.clamp(torch.maximum(dr,dh),max=0)
def sdf_wedge_local(pl,half_size):
    h=half_size.abs()+MIN_SIZE;d=torch.abs(pl)-h
    box=torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv=1.0/torch.sqrt(1.0/(h[1]**2)+1.0/(h[2]**2)+1e-8)
    plane=(pl[:,1]/h[1]+pl[:,2]/h[2]-1.0)*inv
    return torch.maximum(box,plane)
def sdf_corner_wedge_local(pl,half_size):
    h=half_size.abs()+MIN_SIZE;hx,hy,hz=h[0],h[1],h[2]
    d=torch.abs(pl)-h
    box=torch.sqrt(torch.clamp(d,min=0).pow(2).sum(-1)+1e-8)+torch.clamp(d.max(-1)[0],max=0)
    inv1=1.0/torch.sqrt(1.0/(hx**2)+1.0/(hy**2)+1e-8)
    p1=(pl[:,0]/hx+pl[:,1]/hy-1.0)*inv1
    inv2=1.0/torch.sqrt(1.0/(hz**2)+1.0/(hy**2)+1e-8)
    p2=(pl[:,2]/hz+pl[:,1]/hy-1.0)*inv2
    return torch.maximum(torch.maximum(box,p1),p2)
def compute_prim_sdf(ptype,pr,pts):
    if ptype=='block': return sdf_block_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='sphere': return sdf_sphere_local(pts-pr[:3],pr[3])
    if ptype=='cylinder':
        pl=torch.matmul(pts-pr[:3],euler_to_R(pr[5:8]).T)
        return sdf_cylinder_local(pl,pr[3],pr[4])
    if ptype=='wedge': return sdf_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
    if ptype=='corner_wedge': return sdf_corner_wedge_local(torch.matmul(pts-pr[:3],euler_to_R(pr[6:9]).T),pr[3:6])
def _compute_sdf_for_type_k(pl, k, params_k):
    if k==0: return sdf_sphere_local(pl, params_k[0])
    elif k==1: return sdf_block_local(pl, params_k)
    elif k==2: return sdf_cylinder_local(pl, params_k[0], params_k[1])
    elif k==3: return sdf_wedge_local(pl, params_k)
    else: return sdf_corner_wedge_local(pl, params_k)

# BASELINE: discrete overcomplete pool
class DiscretePool(nn.Module):
    PC={'block':9,'sphere':4,'cylinder':8,'wedge':9,'corner_wedge':9}
    def __init__(self,types,init_scale=0.5):
        super().__init__()
        self.types=types;self.num_primitives=len(types)
        total=sum(self.PC[t] for t in types)
        self.params=nn.Parameter(torch.randn(total)*init_scale)
        self.sharpness=nn.Parameter(torch.ones(len(types))*20.0)
        self.param_indices=[];idx=0
        for t in types:
            c=self.PC[t];self.param_indices.append((idx,idx+c));idx+=c
    def get_occ(self,p,i):
        s,e=self.param_indices[i]
        sdf=compute_prim_sdf(self.types[i],self.params[s:e],p)
        return torch.sigmoid(-sdf*self.sharpness[i].abs().clamp(min=1.0))
    def get_all_occs(self,p):
        return torch.stack([self.get_occ(p,i) for i in range(self.num_primitives)],dim=-1)
    def get_center(self,i):
        s,_=self.param_indices[i];return self.params.data[s:s+3]

class BaselineCSG(nn.Module):
    def __init__(self,prims_per_type=16,prim_types=None,num_groups=16,max_group_size=8):
        super().__init__()
        if prim_types is None: prim_types=['block','sphere','cylinder','wedge','corner_wedge']
        all_types=[]
        for t in prim_types: all_types.extend([t]*prims_per_type)
        self.pool=DiscretePool(all_types,0.3)
        self.N=len(all_types);self.K=num_groups;self.max_group_size=max_group_size
        self.m_pos=nn.Parameter(torch.randn(self.N,num_groups)*0.3-3.0)
        self.m_neg=nn.Parameter(torch.randn(self.N,num_groups)*0.3-5.0)
        self.act=nn.Parameter(torch.randn(num_groups)*0.3)
    def forward(self,p,**kw):
        occs=self.pool.get_all_occs(p)
        mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg)
        a=torch.sigmoid(self.act);o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def forward_hard_dnf(self,p,thr=0.5):
        occs=self.pool.get_all_occs(p)
        mp=(torch.sigmoid(self.m_pos)>thr).float();mn=(torch.sigmoid(self.m_neg)>thr).float()
        a=(torch.sigmoid(self.act)>thr).float();o=occs.unsqueeze(-1)
        fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
        lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
        ag=a.unsqueeze(0)*torch.exp(lg)
        return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)
    def structure_loss(self,ls=1e-4,lb=1e-5,lc=1e-4,lgs=1e-5,**kw):
        mp=torch.sigmoid(self.m_pos);mn=torch.sigmoid(self.m_neg);a=torch.sigmoid(self.act)
        sparse=ls*(mp.sum()+mn.sum()+a.sum())
        binary=lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
        conflict=lc*(mp*mn).sum()
        excess=torch.clamp(mp.sum(0)+mn.sum(0)-self.max_group_size,min=0)
        return sparse+binary+conflict+lgs*(excess**2).sum()
    def get_stats(self,thr=0.3):
        mp=torch.sigmoid(self.m_pos).detach();mn=torch.sigmoid(self.m_neg).detach()
        a=torch.sigmoid(self.act).detach()
        used=((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
        ag=(a>thr).sum().item()
        types={}
        for i in range(self.N):
            if mp[i].max()>thr or mn[i].max()>thr:
                t=self.pool.types[i];types[t]=types.get(t,0)+1
        return {'prims_used':used,'groups_active':ag,'type_dist':types,'N':self.N}
    def print_status(self,thr=0.3):
        s=self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")

# BC-TYPE POOL with ALL normalization variants
TYPE_NAMES = ['sphere','block','cylinder','wedge','corner_wedge']
N_TYPES = len(TYPE_NAMES)

class BcTypePool(nn.Module):
    TYPE_INTRINSIC = [1,3,2,3,3]  # 12 total
    def __init__(self,num_primitives,omega=10.0,init_scale=0.3):
        super().__init__()
        self.N=num_primitives; self.omega=omega
        self.centers=nn.Parameter(torch.randn(num_primitives,3)*init_scale)
        self.rotations=nn.Parameter(torch.randn(num_primitives,3)*0.1)
        self.intrinsic=nn.Parameter(torch.randn(num_primitives,sum(self.TYPE_INTRINSIC))*init_scale)
        self.type_logits=nn.Parameter(torch.randn(num_primitives,N_TYPES)*0.3)
        self.sharpness=nn.Parameter(torch.ones(num_primitives)*20.0)

    def _type_offsets(self):
        off=[0]
        for c in self.TYPE_INTRINSIC: off.append(off[-1]+c)
        return off

    def get_type_weights_bc(self,i,temperature=1.0):
        return torch.sigmoid(torch.sin(self.omega*self.type_logits[i])*temperature)

    def _compute_type_sdfs(self,p,i):
        center=self.centers[i]; R=euler_to_R(self.rotations[i])
        pl=torch.matmul(p-center,R.T)
        off=self._type_offsets()
        return [_compute_sdf_for_type_k(pl,k,self.intrinsic[i,off[k]:off[k+1]]) for k in range(N_TYPES)]

    # Scale computations

    def get_type_size_scale(self,i):
        """K: mean absolute parameter value per type."""
        off=self._type_offsets(); scales=[]
        scales.append(self.intrinsic[i,off[0]:off[1]][0].abs()+MIN_SIZE)
        scales.append(self.intrinsic[i,off[1]:off[2]].abs().mean()+MIN_SIZE)
        scales.append(self.intrinsic[i,off[2]:off[3]].abs().mean()+MIN_SIZE)
        scales.append(self.intrinsic[i,off[3]:off[4]].abs().mean()+MIN_SIZE)
        scales.append(self.intrinsic[i,off[4]:off[5]].abs().mean()+MIN_SIZE)
        return scales

    def get_type_vol_radius_scale(self,i):
        """N: volume-equivalent sphere radius r_eff = (3V/4pi)^(1/3)."""
        off=self._type_offsets(); p=self.intrinsic[i]; scales=[]
        # sphere: r_eff = r
        scales.append(p[off[0]:off[1]][0].abs()+MIN_SIZE)
        # block: V=8abc, r_eff=(6abc/pi)^(1/3)
        abc=(p[off[1]:off[2]].abs()+MIN_SIZE)
        scales.append((6.0*abc.prod()/3.14159).pow(1.0/3.0))
        # cylinder: V=2pi*r^2*h, r_eff=(1.5*r^2*h)^(1/3)
        cr=p[off[2]:off[3]][0].abs()+MIN_SIZE
        ch=p[off[2]:off[3]][1].abs()+MIN_SIZE
        scales.append((1.5*cr*cr*ch).pow(1.0/3.0))
        # wedge: V~4abc, r_eff=(3abc/pi)^(1/3)
        wabc=(p[off[3]:off[4]].abs()+MIN_SIZE)
        scales.append((3.0*wabc.prod()/3.14159).pow(1.0/3.0))
        # corner_wedge: V~2abc, r_eff=(1.5abc/pi)^(1/3)
        cwabc=(p[off[4]:off[5]].abs()+MIN_SIZE)
        scales.append((1.5*cwabc.prod()/3.14159).pow(1.0/3.0))
        return scales

    def get_type_area_scale(self,i):
        """O: sqrt(surface area) per type."""
        off=self._type_offsets(); p=self.intrinsic[i]; scales=[]
        # sphere: S=4pi*r^2, sqrt(S)=2r*sqrt(pi)
        r_s=p[off[0]:off[1]][0].abs()+MIN_SIZE
        scales.append(2.0*r_s*1.7725)
        # block: S=8(ab+bc+ac)
        a,b,c=(p[off[1]:off[2]].abs()+MIN_SIZE)
        scales.append(torch.sqrt(8.0*(a*b+b*c+a*c)+1e-8))
        # cylinder: S=2pi*r*(2h+r)
        cr=p[off[2]:off[3]][0].abs()+MIN_SIZE
        ch=p[off[2]:off[3]][1].abs()+MIN_SIZE
        scales.append(torch.sqrt(6.2832*cr*(2*ch+cr)+1e-8))
        # wedge: ~0.75 box
        wa,wb,wc=(p[off[3]:off[4]].abs()+MIN_SIZE)
        scales.append(torch.sqrt(6.0*(wa*wb+wb*wc+wa*wc)+1e-8))
        # corner_wedge: ~0.6 box
        ca,cb,cc=(p[off[4]:off[5]].abs()+MIN_SIZE)
        scales.append(torch.sqrt(5.0*(ca*cb+cb*cc+ca*cc)+1e-8))
        return scales

    def _compute_grad_norms(self,p,i):
        """J: local |grad_x SDF_k| per type, detached."""
        with torch.enable_grad():
            p_det=p.detach().requires_grad_(True)
            center=self.centers[i].detach()
            R=euler_to_R(self.rotations[i].detach())
            pl=torch.matmul(p_det-center,R.T)
            off=self._type_offsets()
            intrinsic_det=self.intrinsic[i].detach()
            grad_norms=[]
            for k in range(N_TYPES):
                sdf=_compute_sdf_for_type_k(pl,k,intrinsic_det[off[k]:off[k+1]])
                grad=torch.autograd.grad(sdf.sum(),p_det,create_graph=False,retain_graph=True)[0]
                grad_norms.append(grad.norm(dim=1).clamp(min=0.01).detach())
        return grad_norms

    # Occupancy modes

    def get_occ_bc(self,p,i,temperature=1.0):
        """G: vanilla Bc product."""
        sdfs=self._compute_type_sdfs(p,i)
        sharp=self.sharpness[i].abs().clamp(min=1.0)
        weights=self.get_type_weights_bc(i,temperature)
        log_occ=torch.zeros(len(p),device=p.device)
        for k in range(N_TYPES):
            occ_k=torch.sigmoid(-sdfs[k]*sharp)
            log_occ=log_occ+torch.log((1.0-weights[k]*occ_k).clamp(min=1e-10))
        return 1.0-torch.exp(log_occ)

    def get_occ_bc_size_norm(self,p,i,temperature=1.0):
        """K: Bc + mean-param size normalization."""
        sdfs=self._compute_type_sdfs(p,i)
        size_scales=self.get_type_size_scale(i)
        sharp=self.sharpness[i].abs().clamp(min=1.0)
        weights=self.get_type_weights_bc(i,temperature)
        log_occ=torch.zeros(len(p),device=p.device)
        for k in range(N_TYPES):
            occ_k=torch.sigmoid(-sdfs[k]/size_scales[k]*sharp)
            log_occ=log_occ+torch.log((1.0-weights[k]*occ_k).clamp(min=1e-10))
        return 1.0-torch.exp(log_occ)

    def get_occ_bc_vol_norm(self,p,i,temperature=1.0):
        """N: Bc + volume-radius normalization."""
        sdfs=self._compute_type_sdfs(p,i)
        vol_scales=self.get_type_vol_radius_scale(i)
        sharp=self.sharpness[i].abs().clamp(min=1.0)
        weights=self.get_type_weights_bc(i,temperature)
        log_occ=torch.zeros(len(p),device=p.device)
        for k in range(N_TYPES):
            occ_k=torch.sigmoid(-sdfs[k]/vol_scales[k]*sharp)
            log_occ=log_occ+torch.log((1.0-weights[k]*occ_k).clamp(min=1e-10))
        return 1.0-torch.exp(log_occ)

    def get_occ_bc_area_norm(self,p,i,temperature=1.0):
        """O: Bc + surface-area normalization."""
        sdfs=self._compute_type_sdfs(p,i)
        area_scales=self.get_type_area_scale(i)
        sharp=self.sharpness[i].abs().clamp(min=1.0)
        weights=self.get_type_weights_bc(i,temperature)
        log_occ=torch.zeros(len(p),device=p.device)
        for k in range(N_TYPES):
            occ_k=torch.sigmoid(-sdfs[k]/area_scales[k]*sharp)
            log_occ=log_occ+torch.log((1.0-weights[k]*occ_k).clamp(min=1e-10))
        return 1.0-torch.exp(log_occ)

    def get_occ_bc_kj_norm(self,p,i,temperature=1.0):
        """P: K+J stacked: size * local-grad normalization."""
        sdfs=self._compute_type_sdfs(p,i)
        size_scales=self.get_type_size_scale(i)
        grad_norms=self._compute_grad_norms(p,i)
        sharp=self.sharpness[i].abs().clamp(min=1.0)
        weights=self.get_type_weights_bc(i,temperature)
        log_occ=torch.zeros(len(p),device=p.device)
        for k in range(N_TYPES):
            occ_k=torch.sigmoid(-sdfs[k]/(size_scales[k]*grad_norms[k])*sharp)
            log_occ=log_occ+torch.log((1.0-weights[k]*occ_k).clamp(min=1e-10))
        return 1.0-torch.exp(log_occ)
    
    def get_occ_attn_gate(self, p, i, temperature=1.0, gamma=1.0):
        """Input-dependent attention-weighted type selection (baseline).

        q_t(x) = softmax_t( log a_t + gamma * phi_t(x) )
        S_i(x) = sum_t q_t(x) * phi_t(x)

        a_t comes from the SAME sine-softmax global weights used by
        bc_vol_norm, so the baseline differs from bc_vol_norm only in
        (a) gate structure (additive-spatial vs multiplicative-through-
        aggregator) and (b) aggregator (sum vs product-union).

        phi_t uses volume-radius normalization at parity with bc_vol_norm.
        """
        sdfs = self._compute_type_sdfs(p, i)
        vol_scales = self.get_type_vol_radius_scale(i)
        sharp = self.sharpness[i].abs().clamp(min=1.0)

        # Volume-normalized local occupancies: (N_TYPES, |p|)
        occs_per_type = torch.stack(
            [torch.sigmoid(-sdfs[k] / vol_scales[k] * sharp)
             for k in range(N_TYPES)], dim=0)

        # Global type preference from the SAME sine-softmax gate used
        # by bc_vol_norm. Normalize to a distribution, take log as the
        # gate logit. This gives the intended baseline:
        #   q_t(x) = softmax_t( log a_t + gamma * phi_t(x) )
        global_weights = self.get_type_weights_bc(i, temperature)
        global_weights = global_weights / (global_weights.sum() + 1e-8)
        global_logits = torch.log(global_weights.clamp_min(1e-8))

        # Per-point gate logits: (N_TYPES, 1) + (N_TYPES, |p|)
        gate_logits = global_logits.unsqueeze(-1) + gamma * occs_per_type
        gate = torch.softmax(gate_logits, dim=0)

        # Additive aggregation
        return (gate * occs_per_type).sum(dim=0)


    def get_occ_attn_gate_hard_type(self, p, i, temperature=20.0):
        """Hard-type extraction for attn_gate.

        Picks ONE primitive type per slot (argmax over sine-softmax
        global weights, at high temperature). Evaluates the slot as
        just that type's volume-normalized occupancy, with no spatial
        gate mixing, so each slot commits to one global type and the hard
        IoU is comparable with the other modes.
        """
        sdfs = self._compute_type_sdfs(p, i)
        vol_scales = self.get_type_vol_radius_scale(i)
        sharp = self.sharpness[i].abs().clamp(min=1.0)

        global_weights = self.get_type_weights_bc(i, temperature)
        t = int(torch.argmax(global_weights).item())

        return torch.sigmoid(-sdfs[t] / vol_scales[t] * sharp)

    # Dispatcher

    def get_all_occs(self, p, mode='bc', temperature=1.0):
        if mode == 'bc':              fn = self.get_occ_bc
        elif mode == 'bc_size_norm':  fn = self.get_occ_bc_size_norm
        elif mode == 'bc_vol_norm':   fn = self.get_occ_bc_vol_norm
        elif mode == 'bc_area_norm':  fn = self.get_occ_bc_area_norm
        elif mode == 'bc_kj_norm':    fn = self.get_occ_bc_kj_norm
        elif mode == 'attn_gate':     fn = self.get_occ_attn_gate
        else:                          fn = self.get_occ_bc
        return torch.stack([fn(p, i, temperature) for i in range(self.N)],
                           dim=-1)
    # Info

    def get_center(self,i): return self.centers.data[i]
    def get_dominant_type(self,i,temperature=20.0):
        weights=self.get_type_weights_bc(i,temperature).detach()
        return TYPE_NAMES[weights.argmax().item()]
    def type_entropy(self,temperature=1.0):
        logits=self.type_logits.detach()
        weights=torch.sigmoid(torch.sin(self.omega*logits)*max(temperature,0.01))
        w_norm=weights/(weights.sum(dim=1,keepdim=True)+1e-8)
        entropy=-(w_norm*torch.log(w_norm+1e-8)).sum(dim=1)
        return entropy.mean().item()

# DNF AGGREGATION (shared)
def _dnf_forward(occs,m_pos,m_neg,act):
    mp=torch.sigmoid(m_pos);mn=torch.sigmoid(m_neg);a=torch.sigmoid(act);o=occs.unsqueeze(-1)
    fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
    lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
    ag=a.unsqueeze(0)*torch.exp(lg)
    return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

def _dnf_forward_hard(occs,m_pos,m_neg,act,thr=0.5):
    mp=(torch.sigmoid(m_pos)>thr).float();mn=(torch.sigmoid(m_neg)>thr).float()
    a=(torch.sigmoid(act)>thr).float();o=occs.unsqueeze(-1)
    fp=1.0-mp.unsqueeze(0)*(1.0-o);fn=1.0-mn.unsqueeze(0)*o
    lg=torch.log(fp.clamp(min=1e-10)).sum(1)+torch.log(fn.clamp(min=1e-10)).sum(1)
    ag=a.unsqueeze(0)*torch.exp(lg)
    return (1.0-(1.0-ag.clamp(0,1)).prod(dim=1)).clamp(0,1)

def _base_structure_loss(m_pos,m_neg,act,max_gs,ls=1e-4,lb=1e-5,lc=1e-4,lgs=1e-5):
    mp=torch.sigmoid(m_pos);mn=torch.sigmoid(m_neg);a=torch.sigmoid(act)
    sparse=ls*(mp.sum()+mn.sum()+a.sum())
    binary=lb*((mp*(1-mp)).sum()+(mn*(1-mn)).sum())
    conflict=lc*(mp*mn).sum()
    excess=torch.clamp(mp.sum(0)+mn.sum(0)-max_gs,min=0)
    return sparse+binary+conflict+lgs*(excess**2).sum()

def _type_structure_loss(type_logits,omega,temperature,lt=5e-5):
    tw=torch.sigmoid(torch.sin(omega*type_logits)*temperature)
    return lt*tw.sum()+lt*(tw*(1-tw)).sum()

def _get_bc_stats(pool,m_pos,m_neg,act,N,K,thr=0.3):
    mp=torch.sigmoid(m_pos).detach();mn=torch.sigmoid(m_neg).detach()
    a=torch.sigmoid(act).detach()
    used=((mp.max(1)[0]>thr)|(mn.max(1)[0]>thr)).sum().item()
    ag=(a>thr).sum().item()
    types={}
    for i in range(N):
        if mp[i].max()>thr or mn[i].max()>thr:
            t=pool.get_dominant_type(i);types[t]=types.get(t,0)+1
    return {'prims_used':used,'groups_active':ag,'type_dist':types,
            'type_entropy':pool.type_entropy(),'N':N}

# GENERIC Bc VARIANT MODEL (handles all modes: G, K, N, O, P)
class BcVariantCSG(nn.Module):
    def __init__(self,num_primitives=50,num_groups=16,omega=10.0,
                 max_group_size=8,occ_mode='bc'):
        super().__init__()
        self.pool=BcTypePool(num_primitives,omega)
        self.N=num_primitives;self.K=num_groups
        self.max_group_size=max_group_size;self.occ_mode=occ_mode
        self.m_pos=nn.Parameter(torch.randn(self.N,num_groups)*0.3-3.0)
        self.m_neg=nn.Parameter(torch.randn(self.N,num_groups)*0.3-5.0)
        self.act=nn.Parameter(torch.randn(num_groups)*0.3)
    def forward(self,p,temperature=1.0,**kw):
        occs=self.pool.get_all_occs(p,mode=self.occ_mode,temperature=temperature)
        return _dnf_forward(occs,self.m_pos,self.m_neg,self.act)
    def forward_hard_dnf(self, p, thr=0.5):
        # For attn_gate, force hard-type extraction: each slot commits
        # to one global primitive type before DNF projection. Other
        # modes' gates already push to one type via sine-softmax at
        # high temperature, so soft-with-temp=20 is already hard enough.
        if self.occ_mode == 'attn_gate':
            occs = torch.stack([
                self.pool.get_occ_attn_gate_hard_type(p, i, temperature=20.0)
                for i in range(self.N)
            ], dim=-1)
        else:
            occs = self.pool.get_all_occs(p, mode=self.occ_mode,
                                            temperature=20.0)
        return _dnf_forward_hard(occs, self.m_pos, self.m_neg, self.act, thr)
    def structure_loss(self,ls=1e-4,lb=1e-5,lc=1e-4,lgs=1e-5,
                       lt=5e-5,temperature=1.0,**kw):
        base=_base_structure_loss(self.m_pos,self.m_neg,self.act,self.max_group_size,ls,lb,lc,lgs)
        ts=_type_structure_loss(self.pool.type_logits,self.pool.omega,temperature,lt)
        return base+ts
    def get_stats(self,thr=0.3):
        return _get_bc_stats(self.pool,self.m_pos,self.m_neg,self.act,self.N,self.K,thr)
    def print_status(self,thr=0.3):
        s=self.get_stats(thr)
        print(f"  Prims: {s['prims_used']}/{self.N}, Groups: {s['groups_active']}/{self.K}")
        print(f"  Types: {s['type_dist']}")
        print(f"  Type entropy: {s['type_entropy']:.3f} (0=committed, 1.6=uniform)")

# INITIALIZATION
def surface_init(model, mesh_sdf):
    N=model.N
    pts,_=trimesh.sample.sample_surface(mesh_sdf.mesh,N*3)
    sel=[0];dists=np.full(len(pts),np.inf)
    for _ in range(N-1):
        d=np.linalg.norm(pts-pts[sel[-1]],axis=1)
        dists=np.minimum(dists,d);sel.append(dists.argmax())
    bbox=mesh_sdf.mesh.bounding_box.extents.max();ts=bbox/(N**(1/3))
    with torch.no_grad():
        pool=model.pool
        if hasattr(pool,'centers'):
            for i in range(N):
                pool.centers.data[i]=torch.from_numpy(pts[sel[i]].astype(np.float32))
                pool.intrinsic.data[i,0]=ts*0.5
                pool.intrinsic.data[i,1:4]=ts*0.4
                pool.intrinsic.data[i,4]=ts*0.3;pool.intrinsic.data[i,5]=ts*0.5
                pool.intrinsic.data[i,6:9]=ts*0.4;pool.intrinsic.data[i,9:12]=ts*0.4
        else:
            for i in range(N):
                s,_=pool.param_indices[i]
                pool.params.data[s:s+3]=torch.from_numpy(pts[sel[i]].astype(np.float32))
                t=pool.types[i]
                if t=='sphere':pool.params.data[s+3]=ts*0.5
                elif t=='cylinder':pool.params.data[s+3]=ts*0.3;pool.params.data[s+4]=ts*0.5
                elif t in ('block','wedge','corner_wedge'):
                    for j in range(3):pool.params.data[s+3+j]=ts*0.4
def spatial_group_init(model, mesh_sdf):
    try:from scipy.cluster.vq import kmeans2
    except:return
    centers=np.array([model.pool.get_center(i).cpu().numpy() for i in range(model.N)])
    nc=min(model.K,model.N//2)
    _,labels=kmeans2(centers,nc,minit='++')
    with torch.no_grad():
        for i in range(model.N):model.m_pos.data[i,labels[i]]=-1.0
        for k in range(nc):
            if (labels==k).sum()>0:model.act.data[k]=1.0

# CHECKPOINTING + TRAINING
@dataclass
class TState:
    epoch:int=0;losses:List[float]=field(default_factory=list)
    ious:List[float]=field(default_factory=list);best_iou:float=0.0;best_epoch:int=0
    hard_ious:List[float]=field(default_factory=list)  # per-epoch hard IoU (CSV log)

class CkptMgr:
    def __init__(self,folder):
        self.folder=folder;os.makedirs(folder,exist_ok=True)
        self.ckpt=os.path.join(folder,'checkpoint.pt')
        self.best=os.path.join(folder,'best_model.pt')
    def save(self,model,optim,state):
        torch.save({'model':model.state_dict(),'optim':optim.state_dict(),
            'state':{'epoch':state.epoch,'losses':state.losses[-5000:],
            'ious':state.ious,'best_iou':state.best_iou,'best_epoch':state.best_epoch}},self.ckpt)
        if state.ious and state.ious[-1]>=state.best_iou:
            torch.save(model.state_dict(),self.best)
    def load(self,model,optim=None):
        if not os.path.exists(self.ckpt):return None
        ck=torch.load(self.ckpt,map_location=device,weights_only=False)
        try:model.load_state_dict(ck['model'])
        except RuntimeError as e:print(f"  ⚠ Ckpt mismatch: {e}");return None
        if optim:
            try:optim.load_state_dict(ck['optim'])
            except:pass
        return TState(**ck['state'])
    def load_best(self,model):
        if os.path.exists(self.best):
            model.load_state_dict(torch.load(self.best,map_location=device,weights_only=True));return True
        return False

def train(model,target_func,mesh_sdf,epochs,output_folder,
          lr=1e-3,num_samples=100000,resample_every=10,
          sparsity_warmup=3000,sparsity_rampup=5000,
          ls=2e-4,lb=5e-5,lc=2e-4,lgs=1e-5,
          ba_start=15000,ba_end=40000,ba_max=5e-4,
          eval_every=200,save_every=2000,resolution=128,
          resume=True,desc="Training",
          use_temperature=False,temp_start=1.0,temp_end=20.0):
    os.makedirs(output_folder,exist_ok=True)
    ckpt=CkptMgr(output_folder);loss_fn=nn.MSELoss()
    _train_t0=time.time()  # per-epoch hard IoU (CSV log)
    eval_grid=sample_3D_grid(resolution);eval_gt=target_func(eval_grid)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=lr*0.01)
    state=None
    if resume:
        state=ckpt.load(model,optimizer)
        if state and state.epoch>=epochs:
            print(f"  Already trained ({state.epoch} ep, best={state.best_iou:.4f})")
            ckpt.load_best(model);return state,eval_grid,eval_gt
        if state:print(f"  Resuming from epoch {state.epoch}")
    if state is None:state=TState()
    P,gt=None,None
    pbar=tqdm(range(state.epoch,epochs),initial=state.epoch,total=epochs,desc=f"  {desc}",leave=False)
    for epoch in pbar:
        progress=epoch/max(epochs-1,1)
        temperature=temp_start+(temp_end-temp_start)*progress if use_temperature else 1.0
        if epoch%resample_every==0 or P is None:
            if mesh_sdf is not None:
                pts,occ=mesh_sdf.sample_points(num_samples)
                P=torch.from_numpy(pts).to(device);gt=torch.from_numpy(occ).to(device)
            else:
                P=torch.rand(num_samples,3,device=device)*2-1;gt=target_func(P)
        pred=model(P,temperature=temperature)
        recon=loss_fn(pred,gt)
        if epoch<sparsity_warmup:sw=0.0
        elif epoch<sparsity_warmup+sparsity_rampup:sw=(epoch-sparsity_warmup)/sparsity_rampup
        else:sw=1.0
        if epoch<ba_start:bm=1.0
        elif epoch<ba_end:bm=1.0+(epoch-ba_start)/(ba_end-ba_start)*(ba_max/max(lb,1e-10)-1.0)
        else:bm=ba_max/max(lb,1e-10)
        struct=model.structure_loss(ls=ls*sw,lb=lb*bm*sw,lc=lc*sw,lgs=lgs*sw,temperature=temperature)
        loss=recon+struct
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();scheduler.step()
        state.epoch=epoch+1;state.losses.append(loss.item())
        if (epoch+1)%eval_every==0:
            eval_temp=temp_end if use_temperature else 1.0
            with torch.no_grad():
                pf=batched_forward(model,eval_grid,temperature=eval_temp)
                iou=compute_iou(pf,eval_gt)
                # Hard IoU at every evaluation
                if hasattr(model,'forward_hard_dnf'):
                    ph=torch.cat([model.forward_hard_dnf(b) for b in torch.split(eval_grid,32**3)],0)
                    hard_iou=compute_iou(ph,eval_gt)
                else:
                    hard_iou=compute_iou((pf>0.5).float(),eval_gt)
                mse=((pf-eval_gt)**2).mean().item()
            state.ious.append(iou)
            state.hard_ious.append(hard_iou)
            if iou>state.best_iou:
                state.best_iou=iou;state.best_epoch=epoch+1
                torch.save(model.state_dict(),ckpt.best)
            s=model.get_stats()
            # Append a row to the per-epoch CSV
            _wall_s=time.time()-_train_t0 if '_train_t0' in dir() else 0.0
            _csv_path=os.path.join(output_folder,'train_log.csv')
            _new_csv=not os.path.exists(_csv_path)
            with open(_csv_path,'a') as _csvf:
                if _new_csv:
                    _csvf.write('epoch,soft_iou,hard_iou,mse,loss,prims_used,groups_active,best_iou,wall_s\n')
                _csvf.write(f'{epoch+1},{iou:.6f},{hard_iou:.6f},{mse:.8f},{recon.item():.6f},{s["prims_used"]},{s["groups_active"]},{state.best_iou:.6f},{_wall_s:.1f}\n')
            pbar.set_postfix(loss=f'{recon.item():.4f}',IoU=f'{iou:.3f}',
                hIoU=f'{hard_iou:.3f}',
                best=f'{state.best_iou:.3f}',g=s['groups_active'],p=s['prims_used'])
        if (epoch+1)%save_every==0:ckpt.save(model,optimizer,state)
    ckpt.save(model,optimizer,state);ckpt.load_best(model)
    return state,eval_grid,eval_gt

def full_eval(model,eval_grid,eval_gt,output_folder,resolution,label="",eval_temperature=1.0):
    os.makedirs(output_folder,exist_ok=True)
    pred=batched_forward(model,eval_grid,temperature=eval_temperature)
    iou=compute_iou(pred,eval_gt);mse=((pred-eval_gt)**2).mean().item()
    fs=compute_fscore(pred,eval_gt)
    mc_from_occ(os.path.join(output_folder,'mesh_soft.obj'),pred,resolution)
    dnf_iou=0.0
    if hasattr(model,'forward_hard_dnf'):
        with torch.no_grad():
            dp=torch.cat([model.forward_hard_dnf(b) for b in torch.split(eval_grid,32**3)],0)
        dnf_iou=compute_iou(dp,eval_gt)
        mc_from_occ(os.path.join(output_folder,'mesh_hard_dnf.obj'),dp,resolution)
    stats=model.get_stats() if hasattr(model,'get_stats') else {}
    results={'iou':iou,'dnf_iou':dnf_iou,'mse':mse,'fscore':fs}
    results.update({k:v for k,v in stats.items() if not isinstance(v,(list,dict))})
    if 'type_dist' in stats:results['type_dist']=stats['type_dist']
    with open(os.path.join(output_folder,'results.json'),'w') as f:
        json.dump(results,f,indent=2)
    print(f"  {label}: IoU={iou:.4f}, DNF={dnf_iou:.4f}, MSE={mse:.6f}, "
          f"F={fs:.4f}, prims={stats.get('prims_used','?')}, groups={stats.get('groups_active','?')}")
    return results

# MAIN
def main():
    parser=argparse.ArgumentParser(description='V7: Normalization Ablations')
    parser.add_argument('--mesh',type=str,default=None)
    parser.add_argument('--epochs',type=int,default=30000)
    parser.add_argument('--baseline-prims',type=int,default=12)
    parser.add_argument('--bc-prims',type=int,default=50)
    parser.add_argument('--groups',type=int,default=16)
    parser.add_argument('--samples',type=int,default=100000)
    parser.add_argument('--resolution',type=int,default=128)
    parser.add_argument('--output',type=str,default='./benchmark_v7/')
    parser.add_argument('--no-resume',action='store_true')
    parser.add_argument('--omega',type=float,default=10.0)
    args=parser.parse_args()
    resume=not args.no_resume

    mesh_sdf=None
    if args.mesh and os.path.exists(args.mesh):
        mesh_sdf=MeshSDF(args.mesh)
        target_func=lambda p:torch.from_numpy(
            (mesh_sdf.query(p.cpu().numpy())<0).astype(np.float32)).to(device)
        target_name=os.path.splitext(os.path.basename(args.mesh))[0]
    else:
        print("Need --mesh"); return

    OUT=os.path.join(args.output,target_name);os.makedirs(OUT,exist_ok=True)
    n_baseline=args.baseline_prims*5

    print(f"\n{'═'*65}")
    print(f"  V7: NORMALIZATION ABLATIONS — {target_name}")
    print(f"{'═'*65}")
    print(f"  A) Baseline: {n_baseline} prims ({args.baseline_prims}×5 types)")
    print(f"  G) Bc-type:  {args.bc_prims} prims (vanilla Bc)")
    print(f"  K) Bc-size:  {args.bc_prims} prims (mean-param normalization)")
    print(f"  N) Bc-vol:   {args.bc_prims} prims (volume-radius normalization)")
    print(f"  O) Bc-area:  {args.bc_prims} prims (surface-area normalization)")
    print(f"  P) Bc-K+J:   {args.bc_prims} prims (size * local-grad, stacked)")
    print(f"  Groups: {args.groups}, Epochs: {args.epochs}, ω={args.omega}")
    print(f"  Device: {device}")
    print(f"{'═'*65}\n")

    eval_grid=sample_3D_grid(args.resolution)
    eval_gt=target_func(eval_grid)
    if mesh_sdf:
        mc_from_sdf(os.path.join(OUT,'00_gt.obj'),mesh_sdf.query_sdf_grid(args.resolution),args.resolution)

    reg=dict(sparsity_warmup=max(args.epochs//20,1500),
             sparsity_rampup=max(args.epochs//12,2000),
             ls=2e-4,lb=5e-5,lc=2e-4,lgs=1e-5,
             ba_start=args.epochs//4,ba_end=int(args.epochs*0.75),ba_max=5e-4)
    all_results={}

    def run_method(key,model,label,eval_temp=1.0,use_temp=False):
        nonlocal all_results
        if mesh_sdf:surface_init(model,mesh_sdf);spatial_group_init(model,mesh_sdf)
        t0=time.time()
        train(model,target_func,mesh_sdf,args.epochs,os.path.join(OUT,key),
              num_samples=args.samples,resolution=args.resolution,
              resume=resume,desc=label,
              use_temperature=use_temp,temp_start=1.0,temp_end=20.0,**reg)
        res=full_eval(model,eval_grid,eval_gt,os.path.join(OUT,key),
                      args.resolution,label,eval_temperature=eval_temp)
        res['time_s']=time.time()-t0
        all_results[key]=res; model.print_status()

    # A) BASELINE
    print(f"\n{'─'*65}\n  A) BASELINE ({n_baseline} prims)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_a=BaselineCSG(prims_per_type=args.baseline_prims,num_groups=args.groups).to(device)
    run_method('A_baseline',model_a,'BASELINE')

    # G) BC-TYPE (vanilla)
    print(f"\n{'─'*65}\n  G) BC-TYPE ({args.bc_prims} prims)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_g=BcVariantCSG(num_primitives=args.bc_prims,num_groups=args.groups,
                          omega=args.omega,occ_mode='bc').to(device)
    run_method('G_bc_type',model_g,'BC-TYPE',eval_temp=20.0,use_temp=True)

    # K) BC + SIZE NORM (mean-param)
    print(f"\n{'─'*65}\n  K) BC + SIZE NORM ({args.bc_prims} prims)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_k=BcVariantCSG(num_primitives=args.bc_prims,num_groups=args.groups,
                          omega=args.omega,occ_mode='bc_size_norm').to(device)
    run_method('K_bc_size_norm',model_k,'BC-SIZE',eval_temp=20.0,use_temp=True)

    # N) BC + VOLUME-RADIUS NORM
    print(f"\n{'─'*65}\n  N) BC + VOL-RADIUS NORM ({args.bc_prims} prims)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_n=BcVariantCSG(num_primitives=args.bc_prims,num_groups=args.groups,
                          omega=args.omega,occ_mode='bc_vol_norm').to(device)
    run_method('N_bc_vol_norm',model_n,'BC-VOL',eval_temp=20.0,use_temp=True)

    # O) BC + SURFACE-AREA NORM
    print(f"\n{'─'*65}\n  O) BC + AREA NORM ({args.bc_prims} prims)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_o=BcVariantCSG(num_primitives=args.bc_prims,num_groups=args.groups,
                          omega=args.omega,occ_mode='bc_area_norm').to(device)
    run_method('O_bc_area_norm',model_o,'BC-AREA',eval_temp=20.0,use_temp=True)

    # P) BC + K+J STACKED
    print(f"\n{'─'*65}\n  P) BC + K+J STACKED ({args.bc_prims} prims, expensive)\n{'─'*65}")
    torch.manual_seed(42);np.random.seed(42)
    model_p=BcVariantCSG(num_primitives=args.bc_prims,num_groups=args.groups,
                          omega=args.omega,occ_mode='bc_kj_norm').to(device)
    run_method('P_bc_kj_norm',model_p,'BC-K+J',eval_temp=20.0,use_temp=True)

    # COMPARISON
    print(f"\n{'━'*65}")
    print(f"  COMPARISON — {target_name}")
    print(f"{'━'*65}")
    print(f"\n  {'Method':<22} {'Soft IoU':>9} {'DNF IoU':>9} {'MSE':>10} "
          f"{'Prims':>6} {'Groups':>7} {'Time':>7}")
    print(f"  {'─'*72}")
    for m in all_results:
        r=all_results[m]
        print(f"  {m:<22} {r['iou']:>9.4f} {r.get('dnf_iou',0):>9.4f} "
              f"{r['mse']:>10.6f} {r.get('prims_used','?'):>6} "
              f"{r.get('groups_active','?'):>7} {r.get('time_s',0):>6.0f}s")
    print(f"  {'─'*72}")

    def _cmp(a_key,b_key,label,win_msg,lose_msg):
        if a_key in all_results and b_key in all_results:
            d=all_results[a_key]['iou']-all_results[b_key]['iou']
            print(f"    {label}: {d:+.4f} IoU")
            print(f"    → {'✓ '+win_msg if d>0 else '✗ '+lose_msg}")

    print(f"\n  KEY COMPARISONS:")
    _cmp('G_bc_type','A_baseline','G vs A (Bc vs baseline)',
         'Bc type selection beats discrete pool','Discrete pool wins')
    _cmp('K_bc_size_norm','G_bc_type','K vs G (size norm vs vanilla)',
         'Size normalization helps','Size normalization not needed')
    _cmp('N_bc_vol_norm','K_bc_size_norm','N vs K (vol-radius vs param-mean)',
         'Volume-radius beats param-mean','Param-mean sufficient')
    _cmp('O_bc_area_norm','K_bc_size_norm','O vs K (area vs param-mean)',
         'Surface-area beats param-mean','Param-mean sufficient')
    _cmp('P_bc_kj_norm','K_bc_size_norm','P vs K (K+J vs K alone)',
         'Macro+micro normalization stacks','Micro redundant after macro')

    # Best method
    best_key=max(all_results,key=lambda k:all_results[k]['iou'])
    print(f"\n  BEST: {best_key} with IoU={all_results[best_key]['iou']:.4f}")

    print(f"\n  TYPE DISTRIBUTIONS:")
    for key in all_results:
        td=all_results[key].get('type_dist',{})
        ent=all_results[key].get('type_entropy',-1)
        ent_str=f", entropy={ent:.3f}" if ent>=0 else ""
        print(f"    {key:<22} {td}{ent_str}")

    print(f"{'━'*65}")
    with open(os.path.join(OUT,'results_v7.json'),'w') as f:
        json.dump({m:{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)}
                   for m,r in all_results.items()},f,indent=2)
    print(f"\n  Output: {OUT}/")

if __name__=='__main__':
    main()
