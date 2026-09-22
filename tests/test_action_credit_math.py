"""Check local-gradient identities without loading a model or using CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
import torch

class GradientIdentities(unittest.TestCase):
    def test_reverse_kl_and_expected_sampled_gradient(self):
        z=torch.tensor([5.,-1.,-2.],dtype=torch.float64,requires_grad=True)
        lq=torch.tensor([.01,.89,.10],dtype=torch.float64).log()
        lp=z.log_softmax(-1);p=lp.exp();kl=(p*(lp-lq)).sum()
        g=torch.autograd.grad(kl,z)[0]
        exact=p*(lp-lq-kl)
        sampled=(lq-lp)[:,None]*(p[None,:]-torch.eye(3,dtype=torch.float64))
        expectation=(p[:,None]*sampled).sum(0)
        torch.testing.assert_close(g,exact)
        torch.testing.assert_close(g,expectation)

    def test_forward_kl_gradient(self):
        z=torch.tensor([5.,-1.,-2.],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.01,.89,.10],dtype=torch.float64)
        loss=(q*(q.log()-z.log_softmax(-1))).sum()
        torch.testing.assert_close(torch.autograd.grad(loss,z)[0],z.softmax(-1)-q)

if __name__=='__main__':unittest.main()
