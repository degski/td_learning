
// MIT License
//
// Copyright (c) 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/************************************************************************

http://www.incompleteideas.net/

http://www.incompleteideas.net/td-backprop-pseudo-code.text

Nonlinear TD/Backprop pseudo C-code

Written by Allen Bonde Jr. and Rich Sutton in April 1992.
Updated in June and August 1993.
Copyright 1993 GTE Laboratories Incorporated. All rights reserved.
Permission is granted to make copies and changes, with attribution,
for research and educational purposes.

This pseudo-code implements a fully-connected two-adaptive-layer network
learning to predict discounted cumulative outcomes through Temporal
Difference learning, as described in Sutton (1988), Barto et al. (1983),
Tesauro (1992), Anderson (1986), Lin (1992), Dayan (1992), et alia. This
is a straightforward combination of discounted TD(lambda) with
backpropagation (Rumelhart, Hinton, and Williams, 1986). This is vanilla
backprop; not even momentum is used. See Sutton & Whitehead (1993) for
an argument that backprop is not the best structural credit assignment
method to use in conjunction with TD. Discounting can be eliminated for
absorbing problems by setting gamma=1. Eligibility traces can be
eliminated by setting lambda=0. Setting both of these parameters to 0
should give standard backprop except where the input at time time_step has its
target presented at time time_step+1.

This is pseudo code: before it can be run it needs I/O, a random
number generator, library links, and some declarations.  We welcome
extensions by others converting this to immediately usable C code.

The network is structured using simple array data structures as follows:


                               OUTPUT

                             ()  ()  ()  l2[k]
                            /  \/  \/  \       k=0...num_output-1
      output_trace[j][k]   /  w1[j][k]  \
                          /              \
                         ()  ()  ()  ()  ()  l1[j]
                          \              /         j=0...num_hidden
   hidden_trace[i][j][k]   \  w2[i][j]  /
                            \  /\  /\  /
                             ()  ()  ()  l0[i]
                                               i=0...num_imputs
                               INPUT


where l0, l1, and l2 are (arrays holding) the activity levels of the input,
hidden, and output units respectively, w2 and w1 are the first and second
layer weights, and hidden_trace and output_trace are the eligibility traces for the first
and second layers (see Sutton, 1989). Not explicitly shown in the figure
are the biases or threshold weights. The first layer bias is provided by
a dummy nth input unit, and the second layer bias is provided by a dummy
(num-hidden)th hidden unit. The activities of both of these dummy units
are held at a constant value (bias).

In addition to the main program, this file contains 4 routines:

    init_network, which initializes the network data structures.

    response, which does the forward propagation, the computation of all
        unit activities based on the current input and weights.

    td_learn, which does the backpropagation of the TD errors, and updates
        the weights.

    update_eligibilities, which updates the eligibility traces.

These routines do all their communication through the global variables
shown in the diagram above, plus old_y, an array holding a copy of the
last time step's output-layer activities.

For simplicity, all the array dimensions are specified as MAX_UNITS, the
maximum allowed number of units in any layer.  This could of course be
tightened up if memory becomes a problem.

REFERENCES

Anderson, C.W. (1986) Learning and Problem Solving with Multilayer
Connectionist Systems, UMass. PhD dissertation, dept. of Computer and
Information Science, Amherts, MA 01003.

Barto, A.G., Sutton, R.S., & Anderson, C.W. (1983) "Neuron-like adaptive
elements that can solve difficult learning control problems," IEEE
Transactions on Systems, Man, and Cybernetics SMC-13: 834-846.

Dayan, P. "The convergence of TD(lambda) for general lambda,"
Machine Learning 8: 341-362.

Lin, L.-J. (1992) "Self-improving reactive agents based on reinforcement
learning, planning and teaching," Machine Learning 8: 293-322.

Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986) "Learning
internal representations by td_error propagation," in D.E. Rumehart & J.L.
McClelland (Eds.), Parallel Distributed Processing: Explorations in the
Microstructure of Cognition, Volume 1: Foundations, 318-362. Cambridge,
MA: MIT Press.

Sutton, R.S. (1988) "Learning to predict by the methods of temporal
differences," Machine Learning 3: 9-44.

Sutton, R.S. (1989) "Implementation details of the TD(lambda) procedure
for the case of vector predictions and backpropagation," GTE
Laboratories Technical Note TN87-509.1, May 1987, corrected August 1989.
Available via ftp from ftp.gte.com as
/pub/reinforcement-learning/sutton-TD-backprop.ps

Sutton, R.S., Whitehead, S.W. (1993) "Online learning with random
representations," Proceedings of the Tenth National Conference on
Machine Learning, 314-321. Soon to be available via ftp from ftp.gte.com
as /pub/reinforcement-learning/sutton-whitehead-93.ps.Z

Tesauro, G. (1992) "Practical issues in temporal difference learning,"
Machine Learning 8: 257-278.
************************************************************************/
