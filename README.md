#Copyright 2014 Twitter, Inc.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

Scalable Algebra Library on top of MapReduce

Current implementation includes:
1) Matrix Multiplication: com.twitter.algebra.matrix.multiply.AtB\_DMJ
2) (Sparse) NMF: com.twitter.algebra.nmf.NMFDriver

The framework has support for:
1) Indexed Matrix Format
2) Dynamic Map-side Join
