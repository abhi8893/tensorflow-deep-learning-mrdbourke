??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv2d_76/kernel
}
$conv2d_76/kernel/Read/ReadVariableOpReadVariableOpconv2d_76/kernel*&
_output_shapes
:
*
dtype0
t
conv2d_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_76/bias
m
"conv2d_76/bias/Read/ReadVariableOpReadVariableOpconv2d_76/bias*
_output_shapes
:
*
dtype0
?
conv2d_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_77/kernel
}
$conv2d_77/kernel/Read/ReadVariableOpReadVariableOpconv2d_77/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_77/bias
m
"conv2d_77/bias/Read/ReadVariableOpReadVariableOpconv2d_77/bias*
_output_shapes
:
*
dtype0
?
conv2d_78/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_78/kernel
}
$conv2d_78/kernel/Read/ReadVariableOpReadVariableOpconv2d_78/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_78/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_78/bias
m
"conv2d_78/bias/Read/ReadVariableOpReadVariableOpconv2d_78/bias*
_output_shapes
:
*
dtype0
?
conv2d_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_79/kernel
}
$conv2d_79/kernel/Read/ReadVariableOpReadVariableOpconv2d_79/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_79/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_79/bias
m
"conv2d_79/bias/Read/ReadVariableOpReadVariableOpconv2d_79/bias*
_output_shapes
:
*
dtype0
?
conv2d_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_80/kernel
}
$conv2d_80/kernel/Read/ReadVariableOpReadVariableOpconv2d_80/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_80/bias
m
"conv2d_80/bias/Read/ReadVariableOpReadVariableOpconv2d_80/bias*
_output_shapes
:
*
dtype0
?
conv2d_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_81/kernel
}
$conv2d_81/kernel/Read/ReadVariableOpReadVariableOpconv2d_81/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_81/bias
m
"conv2d_81/bias/Read/ReadVariableOpReadVariableOpconv2d_81/bias*
_output_shapes
:
*
dtype0
|
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?* 
shared_namedense_20/kernel
u
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel* 
_output_shapes
:
?-?*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:?*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	?
*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/conv2d_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_76/kernel/m
?
+Adam/conv2d_76/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_76/kernel/m*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_76/bias/m
{
)Adam/conv2d_76/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_76/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_77/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_77/kernel/m
?
+Adam/conv2d_77/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/kernel/m*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_77/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_77/bias/m
{
)Adam/conv2d_77/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_78/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_78/kernel/m
?
+Adam/conv2d_78/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_78/kernel/m*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_78/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_78/bias/m
{
)Adam/conv2d_78/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_78/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_79/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_79/kernel/m
?
+Adam/conv2d_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/kernel/m*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_79/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_79/bias/m
{
)Adam/conv2d_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_80/kernel/m
?
+Adam/conv2d_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/kernel/m*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_80/bias/m
{
)Adam/conv2d_80/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_81/kernel/m
?
+Adam/conv2d_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/m*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_81/bias/m
{
)Adam/conv2d_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?*'
shared_nameAdam/dense_20/kernel/m
?
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m* 
_output_shapes
:
?-?*
dtype0
?
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_20/bias/m
z
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/dense_21/kernel/m
?
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_76/kernel/v
?
+Adam/conv2d_76/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_76/kernel/v*&
_output_shapes
:
*
dtype0
?
Adam/conv2d_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_76/bias/v
{
)Adam/conv2d_76/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_76/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv2d_77/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_77/kernel/v
?
+Adam/conv2d_77/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/kernel/v*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_77/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_77/bias/v
{
)Adam/conv2d_77/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv2d_78/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_78/kernel/v
?
+Adam/conv2d_78/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_78/kernel/v*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_78/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_78/bias/v
{
)Adam/conv2d_78/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_78/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv2d_79/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_79/kernel/v
?
+Adam/conv2d_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/kernel/v*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_79/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_79/bias/v
{
)Adam/conv2d_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_79/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv2d_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_80/kernel/v
?
+Adam/conv2d_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/kernel/v*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_80/bias/v
{
)Adam/conv2d_80/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_80/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv2d_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_81/kernel/v
?
+Adam/conv2d_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/kernel/v*&
_output_shapes
:

*
dtype0
?
Adam/conv2d_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_81/bias/v
{
)Adam/conv2d_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_81/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?-?*'
shared_nameAdam/dense_20/kernel/v
?
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v* 
_output_shapes
:
?-?*
dtype0
?
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_20/bias/v
z
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/dense_21/kernel/v
?
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?\
value?\B?\ B?\
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
R
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
R
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
?
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratem?m?m?m?#m?$m?)m?*m?3m?4m?9m?:m?Gm?Hm?Mm?Nm?v?v?v?v?#v?$v?)v?*v?3v?4v?9v?:v?Gv?Hv?Mv?Nv?
 
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15
?
regularization_losses
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_regularization_losses
	variables
trainable_variables
\layer_metrics
 
\Z
VARIABLE_VALUEconv2d_76/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_76/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
]non_trainable_variables
^metrics
_layer_regularization_losses

`layers
	variables
trainable_variables
alayer_metrics
\Z
VARIABLE_VALUEconv2d_77/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_77/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
bnon_trainable_variables
cmetrics
dlayer_regularization_losses

elayers
	variables
trainable_variables
flayer_metrics
 
 
 
?
regularization_losses
gnon_trainable_variables
hmetrics
ilayer_regularization_losses

jlayers
 	variables
!trainable_variables
klayer_metrics
\Z
VARIABLE_VALUEconv2d_78/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_78/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
%regularization_losses
lnon_trainable_variables
mmetrics
nlayer_regularization_losses

olayers
&	variables
'trainable_variables
player_metrics
\Z
VARIABLE_VALUEconv2d_79/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_79/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
+regularization_losses
qnon_trainable_variables
rmetrics
slayer_regularization_losses

tlayers
,	variables
-trainable_variables
ulayer_metrics
 
 
 
?
/regularization_losses
vnon_trainable_variables
wmetrics
xlayer_regularization_losses

ylayers
0	variables
1trainable_variables
zlayer_metrics
\Z
VARIABLE_VALUEconv2d_80/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_80/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
5regularization_losses
{non_trainable_variables
|metrics
}layer_regularization_losses

~layers
6	variables
7trainable_variables
layer_metrics
\Z
VARIABLE_VALUEconv2d_81/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_81/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
?
;regularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
<	variables
=trainable_variables
?layer_metrics
 
 
 
?
?regularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
@	variables
Atrainable_variables
?layer_metrics
 
 
 
?
Cregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
D	variables
Etrainable_variables
?layer_metrics
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
?
Iregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
J	variables
Ktrainable_variables
?layer_metrics
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
?
Oregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
P	variables
Qtrainable_variables
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
?2
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_76/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_76/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_77/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_77/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_78/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_78/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_79/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_79/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_80/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_80/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_81/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_76/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_76/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_77/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_77/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_78/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_78/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_79/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_79/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_80/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_80/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_81/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_81/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_17Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17conv2d_76/kernelconv2d_76/biasconv2d_77/kernelconv2d_77/biasconv2d_78/kernelconv2d_78/biasconv2d_79/kernelconv2d_79/biasconv2d_80/kernelconv2d_80/biasconv2d_81/kernelconv2d_81/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_103666
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_76/kernel/Read/ReadVariableOp"conv2d_76/bias/Read/ReadVariableOp$conv2d_77/kernel/Read/ReadVariableOp"conv2d_77/bias/Read/ReadVariableOp$conv2d_78/kernel/Read/ReadVariableOp"conv2d_78/bias/Read/ReadVariableOp$conv2d_79/kernel/Read/ReadVariableOp"conv2d_79/bias/Read/ReadVariableOp$conv2d_80/kernel/Read/ReadVariableOp"conv2d_80/bias/Read/ReadVariableOp$conv2d_81/kernel/Read/ReadVariableOp"conv2d_81/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/conv2d_76/kernel/m/Read/ReadVariableOp)Adam/conv2d_76/bias/m/Read/ReadVariableOp+Adam/conv2d_77/kernel/m/Read/ReadVariableOp)Adam/conv2d_77/bias/m/Read/ReadVariableOp+Adam/conv2d_78/kernel/m/Read/ReadVariableOp)Adam/conv2d_78/bias/m/Read/ReadVariableOp+Adam/conv2d_79/kernel/m/Read/ReadVariableOp)Adam/conv2d_79/bias/m/Read/ReadVariableOp+Adam/conv2d_80/kernel/m/Read/ReadVariableOp)Adam/conv2d_80/bias/m/Read/ReadVariableOp+Adam/conv2d_81/kernel/m/Read/ReadVariableOp)Adam/conv2d_81/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp+Adam/conv2d_76/kernel/v/Read/ReadVariableOp)Adam/conv2d_76/bias/v/Read/ReadVariableOp+Adam/conv2d_77/kernel/v/Read/ReadVariableOp)Adam/conv2d_77/bias/v/Read/ReadVariableOp+Adam/conv2d_78/kernel/v/Read/ReadVariableOp)Adam/conv2d_78/bias/v/Read/ReadVariableOp+Adam/conv2d_79/kernel/v/Read/ReadVariableOp)Adam/conv2d_79/bias/v/Read/ReadVariableOp+Adam/conv2d_80/kernel/v/Read/ReadVariableOp)Adam/conv2d_80/bias/v/Read/ReadVariableOp+Adam/conv2d_81/kernel/v/Read/ReadVariableOp)Adam/conv2d_81/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_104241
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_76/kernelconv2d_76/biasconv2d_77/kernelconv2d_77/biasconv2d_78/kernelconv2d_78/biasconv2d_79/kernelconv2d_79/biasconv2d_80/kernelconv2d_80/biasconv2d_81/kernelconv2d_81/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv2d_76/kernel/mAdam/conv2d_76/bias/mAdam/conv2d_77/kernel/mAdam/conv2d_77/bias/mAdam/conv2d_78/kernel/mAdam/conv2d_78/bias/mAdam/conv2d_79/kernel/mAdam/conv2d_79/bias/mAdam/conv2d_80/kernel/mAdam/conv2d_80/bias/mAdam/conv2d_81/kernel/mAdam/conv2d_81/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/conv2d_76/kernel/vAdam/conv2d_76/bias/vAdam/conv2d_77/kernel/vAdam/conv2d_77/bias/vAdam/conv2d_78/kernel/vAdam/conv2d_78/bias/vAdam/conv2d_79/kernel/vAdam/conv2d_79/bias/vAdam/conv2d_80/kernel/vAdam/conv2d_80/bias/vAdam/conv2d_81/kernel/vAdam/conv2d_81/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/v*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_104428??	
?8
?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103448
input_17
conv2d_76_103403
conv2d_76_103405
conv2d_77_103408
conv2d_77_103410
conv2d_78_103414
conv2d_78_103416
conv2d_79_103419
conv2d_79_103421
conv2d_80_103425
conv2d_80_103427
conv2d_81_103430
conv2d_81_103432
dense_20_103437
dense_20_103439
dense_21_103442
dense_21_103444
identity??!conv2d_76/StatefulPartitionedCall?!conv2d_77/StatefulPartitionedCall?!conv2d_78/StatefulPartitionedCall?!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCallinput_17conv2d_76_103403conv2d_76_103405*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_76_layer_call_and_return_conditional_losses_1031772#
!conv2d_76/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0conv2d_77_103408conv2d_77_103410*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_77_layer_call_and_return_conditional_losses_1032042#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1031322"
 max_pooling2d_38/PartitionedCall?
!conv2d_78/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_78_103414conv2d_78_103416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_78_layer_call_and_return_conditional_losses_1032322#
!conv2d_78/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_78/StatefulPartitionedCall:output:0conv2d_79_103419conv2d_79_103421*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_79_layer_call_and_return_conditional_losses_1032592#
!conv2d_79/StatefulPartitionedCall?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1031442"
 max_pooling2d_39/PartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_80_103425conv2d_80_103427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????33
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_80_layer_call_and_return_conditional_losses_1032872#
!conv2d_80/StatefulPartitionedCall?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0conv2d_81_103430conv2d_81_103432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????11
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_1033142#
!conv2d_81/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_1031562"
 max_pooling2d_40/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_1033372
flatten_16/PartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_20_103437dense_20_103439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1033562"
 dense_20/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_103442dense_21_103444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_1033832"
 dense_21/StatefulPartitionedCall?
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall"^conv2d_78/StatefulPartitionedCall"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!conv2d_78/StatefulPartitionedCall!conv2d_78/StatefulPartitionedCall2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
?
~
)__inference_dense_20_layer_call_fn_104021

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1033562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????-::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?

*__inference_conv2d_81_layer_call_fn_103990

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????11
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_1033142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????11
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????33
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????33

 
_user_specified_nameinputs
?

?
E__inference_conv2d_81_layer_call_and_return_conditional_losses_103314

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????11
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????11
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????33
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????33

 
_user_specified_nameinputs
?

?
E__inference_conv2d_77_layer_call_and_return_conditional_losses_103901

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????

 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_103666
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1031262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
?

?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103833

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_1034992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_77_layer_call_and_return_conditional_losses_103204

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????

 
_user_specified_nameinputs
?

?
E__inference_conv2d_80_layer_call_and_return_conditional_losses_103287

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????33
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????33
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????55
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????55

 
_user_specified_nameinputs
?
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_103337

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
D__inference_dense_21_layer_call_and_return_conditional_losses_103383

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103400
input_17
conv2d_76_103188
conv2d_76_103190
conv2d_77_103215
conv2d_77_103217
conv2d_78_103243
conv2d_78_103245
conv2d_79_103270
conv2d_79_103272
conv2d_80_103298
conv2d_80_103300
conv2d_81_103325
conv2d_81_103327
dense_20_103367
dense_20_103369
dense_21_103394
dense_21_103396
identity??!conv2d_76/StatefulPartitionedCall?!conv2d_77/StatefulPartitionedCall?!conv2d_78/StatefulPartitionedCall?!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCallinput_17conv2d_76_103188conv2d_76_103190*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_76_layer_call_and_return_conditional_losses_1031772#
!conv2d_76/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0conv2d_77_103215conv2d_77_103217*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_77_layer_call_and_return_conditional_losses_1032042#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1031322"
 max_pooling2d_38/PartitionedCall?
!conv2d_78/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_78_103243conv2d_78_103245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_78_layer_call_and_return_conditional_losses_1032322#
!conv2d_78/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_78/StatefulPartitionedCall:output:0conv2d_79_103270conv2d_79_103272*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_79_layer_call_and_return_conditional_losses_1032592#
!conv2d_79/StatefulPartitionedCall?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1031442"
 max_pooling2d_39/PartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_80_103298conv2d_80_103300*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????33
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_80_layer_call_and_return_conditional_losses_1032872#
!conv2d_80/StatefulPartitionedCall?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0conv2d_81_103325conv2d_81_103327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????11
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_1033142#
!conv2d_81/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_1031562"
 max_pooling2d_40/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_1033372
flatten_16/PartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_20_103367dense_20_103369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1033562"
 dense_20/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_103394dense_21_103396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_1033832"
 dense_21/StatefulPartitionedCall?
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall"^conv2d_78/StatefulPartitionedCall"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!conv2d_78/StatefulPartitionedCall!conv2d_78/StatefulPartitionedCall2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
?8
?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103584

inputs
conv2d_76_103539
conv2d_76_103541
conv2d_77_103544
conv2d_77_103546
conv2d_78_103550
conv2d_78_103552
conv2d_79_103555
conv2d_79_103557
conv2d_80_103561
conv2d_80_103563
conv2d_81_103566
conv2d_81_103568
dense_20_103573
dense_20_103575
dense_21_103578
dense_21_103580
identity??!conv2d_76/StatefulPartitionedCall?!conv2d_77/StatefulPartitionedCall?!conv2d_78/StatefulPartitionedCall?!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_76_103539conv2d_76_103541*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_76_layer_call_and_return_conditional_losses_1031772#
!conv2d_76/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0conv2d_77_103544conv2d_77_103546*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_77_layer_call_and_return_conditional_losses_1032042#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1031322"
 max_pooling2d_38/PartitionedCall?
!conv2d_78/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_78_103550conv2d_78_103552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_78_layer_call_and_return_conditional_losses_1032322#
!conv2d_78/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_78/StatefulPartitionedCall:output:0conv2d_79_103555conv2d_79_103557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_79_layer_call_and_return_conditional_losses_1032592#
!conv2d_79/StatefulPartitionedCall?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1031442"
 max_pooling2d_39/PartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_80_103561conv2d_80_103563*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????33
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_80_layer_call_and_return_conditional_losses_1032872#
!conv2d_80/StatefulPartitionedCall?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0conv2d_81_103566conv2d_81_103568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????11
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_1033142#
!conv2d_81/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_1031562"
 max_pooling2d_40/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_1033372
flatten_16/PartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_20_103573dense_20_103575*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1033562"
 dense_20/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_103578dense_21_103580*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_1033832"
 dense_21/StatefulPartitionedCall?
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall"^conv2d_78/StatefulPartitionedCall"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!conv2d_78/StatefulPartitionedCall!conv2d_78/StatefulPartitionedCall2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_39_layer_call_fn_103150

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1031442
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv2d_77_layer_call_fn_103910

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_77_layer_call_and_return_conditional_losses_1032042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????
::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????

 
_user_specified_nameinputs
?

?
E__inference_conv2d_79_layer_call_and_return_conditional_losses_103259

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????jj
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????jj
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????ll
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????ll

 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_40_layer_call_fn_103162

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_1031562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_103132

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_38_layer_call_fn_103138

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1031322
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103870

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_1035842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_80_layer_call_and_return_conditional_losses_103961

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????33
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????33
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????55
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????55

 
_user_specified_nameinputs
?
~
)__inference_dense_21_layer_call_fn_104041

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_1033832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_76_layer_call_and_return_conditional_losses_103881

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?v
?
__inference__traced_save_104241
file_prefix/
+savev2_conv2d_76_kernel_read_readvariableop-
)savev2_conv2d_76_bias_read_readvariableop/
+savev2_conv2d_77_kernel_read_readvariableop-
)savev2_conv2d_77_bias_read_readvariableop/
+savev2_conv2d_78_kernel_read_readvariableop-
)savev2_conv2d_78_bias_read_readvariableop/
+savev2_conv2d_79_kernel_read_readvariableop-
)savev2_conv2d_79_bias_read_readvariableop/
+savev2_conv2d_80_kernel_read_readvariableop-
)savev2_conv2d_80_bias_read_readvariableop/
+savev2_conv2d_81_kernel_read_readvariableop-
)savev2_conv2d_81_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_conv2d_76_kernel_m_read_readvariableop4
0savev2_adam_conv2d_76_bias_m_read_readvariableop6
2savev2_adam_conv2d_77_kernel_m_read_readvariableop4
0savev2_adam_conv2d_77_bias_m_read_readvariableop6
2savev2_adam_conv2d_78_kernel_m_read_readvariableop4
0savev2_adam_conv2d_78_bias_m_read_readvariableop6
2savev2_adam_conv2d_79_kernel_m_read_readvariableop4
0savev2_adam_conv2d_79_bias_m_read_readvariableop6
2savev2_adam_conv2d_80_kernel_m_read_readvariableop4
0savev2_adam_conv2d_80_bias_m_read_readvariableop6
2savev2_adam_conv2d_81_kernel_m_read_readvariableop4
0savev2_adam_conv2d_81_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_76_kernel_v_read_readvariableop4
0savev2_adam_conv2d_76_bias_v_read_readvariableop6
2savev2_adam_conv2d_77_kernel_v_read_readvariableop4
0savev2_adam_conv2d_77_bias_v_read_readvariableop6
2savev2_adam_conv2d_78_kernel_v_read_readvariableop4
0savev2_adam_conv2d_78_bias_v_read_readvariableop6
2savev2_adam_conv2d_79_kernel_v_read_readvariableop4
0savev2_adam_conv2d_79_bias_v_read_readvariableop6
2savev2_adam_conv2d_80_kernel_v_read_readvariableop4
0savev2_adam_conv2d_80_bias_v_read_readvariableop6
2savev2_adam_conv2d_81_kernel_v_read_readvariableop4
0savev2_adam_conv2d_81_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*? 
value? B? <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_76_kernel_read_readvariableop)savev2_conv2d_76_bias_read_readvariableop+savev2_conv2d_77_kernel_read_readvariableop)savev2_conv2d_77_bias_read_readvariableop+savev2_conv2d_78_kernel_read_readvariableop)savev2_conv2d_78_bias_read_readvariableop+savev2_conv2d_79_kernel_read_readvariableop)savev2_conv2d_79_bias_read_readvariableop+savev2_conv2d_80_kernel_read_readvariableop)savev2_conv2d_80_bias_read_readvariableop+savev2_conv2d_81_kernel_read_readvariableop)savev2_conv2d_81_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_conv2d_76_kernel_m_read_readvariableop0savev2_adam_conv2d_76_bias_m_read_readvariableop2savev2_adam_conv2d_77_kernel_m_read_readvariableop0savev2_adam_conv2d_77_bias_m_read_readvariableop2savev2_adam_conv2d_78_kernel_m_read_readvariableop0savev2_adam_conv2d_78_bias_m_read_readvariableop2savev2_adam_conv2d_79_kernel_m_read_readvariableop0savev2_adam_conv2d_79_bias_m_read_readvariableop2savev2_adam_conv2d_80_kernel_m_read_readvariableop0savev2_adam_conv2d_80_bias_m_read_readvariableop2savev2_adam_conv2d_81_kernel_m_read_readvariableop0savev2_adam_conv2d_81_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop2savev2_adam_conv2d_76_kernel_v_read_readvariableop0savev2_adam_conv2d_76_bias_v_read_readvariableop2savev2_adam_conv2d_77_kernel_v_read_readvariableop0savev2_adam_conv2d_77_bias_v_read_readvariableop2savev2_adam_conv2d_78_kernel_v_read_readvariableop0savev2_adam_conv2d_78_bias_v_read_readvariableop2savev2_adam_conv2d_79_kernel_v_read_readvariableop0savev2_adam_conv2d_79_bias_v_read_readvariableop2savev2_adam_conv2d_80_kernel_v_read_readvariableop0savev2_adam_conv2d_80_bias_v_read_readvariableop2savev2_adam_conv2d_81_kernel_v_read_readvariableop0savev2_adam_conv2d_81_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
:
:

:
:

:
:

:
:

:
:

:
:
?-?:?:	?
:
: : : : : : : : : : : :
:
:

:
:

:
:

:
:

:
:

:
:
?-?:?:	?
:
:
:
:

:
:

:
:

:
:

:
:

:
:
?-?:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,	(
&
_output_shapes
:

: 


_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&"
 
_output_shapes
:
?-?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:, (
&
_output_shapes
:

: !

_output_shapes
:
:,"(
&
_output_shapes
:

: #

_output_shapes
:
:,$(
&
_output_shapes
:

: %

_output_shapes
:
:,&(
&
_output_shapes
:

: '

_output_shapes
:
:&("
 
_output_shapes
:
?-?:!)

_output_shapes	
:?:%*!

_output_shapes
:	?
: +

_output_shapes
:
:,,(
&
_output_shapes
:
: -

_output_shapes
:
:,.(
&
_output_shapes
:

: /

_output_shapes
:
:,0(
&
_output_shapes
:

: 1

_output_shapes
:
:,2(
&
_output_shapes
:

: 3

_output_shapes
:
:,4(
&
_output_shapes
:

: 5

_output_shapes
:
:,6(
&
_output_shapes
:

: 7

_output_shapes
:
:&8"
 
_output_shapes
:
?-?:!9

_output_shapes	
:?:%:!

_output_shapes
:	?
: ;

_output_shapes
:
:<

_output_shapes
: 
?
G
+__inference_flatten_16_layer_call_fn_104001

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_1033372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?W
?

T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103731

inputs,
(conv2d_76_conv2d_readvariableop_resource-
)conv2d_76_biasadd_readvariableop_resource,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource,
(conv2d_78_conv2d_readvariableop_resource-
)conv2d_78_biasadd_readvariableop_resource,
(conv2d_79_conv2d_readvariableop_resource-
)conv2d_79_biasadd_readvariableop_resource,
(conv2d_80_conv2d_readvariableop_resource-
)conv2d_80_biasadd_readvariableop_resource,
(conv2d_81_conv2d_readvariableop_resource-
)conv2d_81_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource
identity?? conv2d_76/BiasAdd/ReadVariableOp?conv2d_76/Conv2D/ReadVariableOp? conv2d_77/BiasAdd/ReadVariableOp?conv2d_77/Conv2D/ReadVariableOp? conv2d_78/BiasAdd/ReadVariableOp?conv2d_78/Conv2D/ReadVariableOp? conv2d_79/BiasAdd/ReadVariableOp?conv2d_79/Conv2D/ReadVariableOp? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_76/Conv2D/ReadVariableOp?
conv2d_76/Conv2DConv2Dinputs'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
conv2d_76/Conv2D?
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_76/BiasAdd/ReadVariableOp?
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2
conv2d_76/BiasAdd?
conv2d_76/ReluReluconv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
conv2d_76/Relu?
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_77/Conv2D/ReadVariableOp?
conv2d_77/Conv2DConv2Dconv2d_76/Relu:activations:0'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
conv2d_77/Conv2D?
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp?
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2
conv2d_77/BiasAdd?
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
conv2d_77/Relu?
max_pooling2d_38/MaxPoolMaxPoolconv2d_77/Relu:activations:0*/
_output_shapes
:?????????nn
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool?
conv2d_78/Conv2D/ReadVariableOpReadVariableOp(conv2d_78_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_78/Conv2D/ReadVariableOp?
conv2d_78/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_78/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
*
paddingVALID*
strides
2
conv2d_78/Conv2D?
 conv2d_78/BiasAdd/ReadVariableOpReadVariableOp)conv2d_78_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_78/BiasAdd/ReadVariableOp?
conv2d_78/BiasAddBiasAddconv2d_78/Conv2D:output:0(conv2d_78/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
2
conv2d_78/BiasAdd~
conv2d_78/ReluReluconv2d_78/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll
2
conv2d_78/Relu?
conv2d_79/Conv2D/ReadVariableOpReadVariableOp(conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_79/Conv2D/ReadVariableOp?
conv2d_79/Conv2DConv2Dconv2d_78/Relu:activations:0'conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
*
paddingVALID*
strides
2
conv2d_79/Conv2D?
 conv2d_79/BiasAdd/ReadVariableOpReadVariableOp)conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_79/BiasAdd/ReadVariableOp?
conv2d_79/BiasAddBiasAddconv2d_79/Conv2D:output:0(conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
2
conv2d_79/BiasAdd~
conv2d_79/ReluReluconv2d_79/BiasAdd:output:0*
T0*/
_output_shapes
:?????????jj
2
conv2d_79/Relu?
max_pooling2d_39/MaxPoolMaxPoolconv2d_79/Relu:activations:0*/
_output_shapes
:?????????55
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_80/Conv2D/ReadVariableOp?
conv2d_80/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
*
paddingVALID*
strides
2
conv2d_80/Conv2D?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_80/BiasAdd/ReadVariableOp?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
2
conv2d_80/BiasAdd~
conv2d_80/ReluReluconv2d_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????33
2
conv2d_80/Relu?
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_81/Conv2D/ReadVariableOp?
conv2d_81/Conv2DConv2Dconv2d_80/Relu:activations:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
*
paddingVALID*
strides
2
conv2d_81/Conv2D?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_81/BiasAdd/ReadVariableOp?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
2
conv2d_81/BiasAdd~
conv2d_81/ReluReluconv2d_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????11
2
conv2d_81/Relu?
max_pooling2d_40/MaxPoolMaxPoolconv2d_81/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPoolu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_16/Const?
flatten_16/ReshapeReshape!max_pooling2d_40/MaxPool:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????-2
flatten_16/Reshape?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMulflatten_16/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_20/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_21/BiasAdd|
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_21/Softmax?
IdentityIdentitydense_21/Softmax:softmax:0!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOp!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp!^conv2d_78/BiasAdd/ReadVariableOp ^conv2d_78/Conv2D/ReadVariableOp!^conv2d_79/BiasAdd/ReadVariableOp ^conv2d_79/Conv2D/ReadVariableOp!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2D
 conv2d_78/BiasAdd/ReadVariableOp conv2d_78/BiasAdd/ReadVariableOp2B
conv2d_78/Conv2D/ReadVariableOpconv2d_78/Conv2D/ReadVariableOp2D
 conv2d_79/BiasAdd/ReadVariableOp conv2d_79/BiasAdd/ReadVariableOp2B
conv2d_79/Conv2D/ReadVariableOpconv2d_79/Conv2D/ReadVariableOp2D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_103144

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_20_layer_call_and_return_conditional_losses_103356

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?

?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103534
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_1034992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
??
?
!__inference__wrapped_model_103126
input_17E
Atinyvgg_extra_conv_dense_conv2d_76_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_76_biasadd_readvariableop_resourceE
Atinyvgg_extra_conv_dense_conv2d_77_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_77_biasadd_readvariableop_resourceE
Atinyvgg_extra_conv_dense_conv2d_78_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_78_biasadd_readvariableop_resourceE
Atinyvgg_extra_conv_dense_conv2d_79_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_79_biasadd_readvariableop_resourceE
Atinyvgg_extra_conv_dense_conv2d_80_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_80_biasadd_readvariableop_resourceE
Atinyvgg_extra_conv_dense_conv2d_81_conv2d_readvariableop_resourceF
Btinyvgg_extra_conv_dense_conv2d_81_biasadd_readvariableop_resourceD
@tinyvgg_extra_conv_dense_dense_20_matmul_readvariableop_resourceE
Atinyvgg_extra_conv_dense_dense_20_biasadd_readvariableop_resourceD
@tinyvgg_extra_conv_dense_dense_21_matmul_readvariableop_resourceE
Atinyvgg_extra_conv_dense_dense_21_biasadd_readvariableop_resource
identity??9TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp?9TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp?9TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp?9TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp?9TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp?9TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp?7TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp?8TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp?7TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp?
8TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2DConv2Dinput_17@TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2,
*TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_76/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2)
'TinyVGG-Extra-Conv-Dense/conv2d_76/Relu?
8TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2DConv2D5TinyVGG-Extra-Conv-Dense/conv2d_76/Relu:activations:0@TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_77_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2,
*TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_77/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2)
'TinyVGG-Extra-Conv-Dense/conv2d_77/Relu?
1TinyVGG-Extra-Conv-Dense/max_pooling2d_38/MaxPoolMaxPool5TinyVGG-Extra-Conv-Dense/conv2d_77/Relu:activations:0*/
_output_shapes
:?????????nn
*
ksize
*
paddingVALID*
strides
23
1TinyVGG-Extra-Conv-Dense/max_pooling2d_38/MaxPool?
8TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_78_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2DConv2D:TinyVGG-Extra-Conv-Dense/max_pooling2d_38/MaxPool:output:0@TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_78_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
2,
*TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_78/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll
2)
'TinyVGG-Extra-Conv-Dense/conv2d_78/Relu?
8TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2DConv2D5TinyVGG-Extra-Conv-Dense/conv2d_78/Relu:activations:0@TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
2,
*TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_79/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd:output:0*
T0*/
_output_shapes
:?????????jj
2)
'TinyVGG-Extra-Conv-Dense/conv2d_79/Relu?
1TinyVGG-Extra-Conv-Dense/max_pooling2d_39/MaxPoolMaxPool5TinyVGG-Extra-Conv-Dense/conv2d_79/Relu:activations:0*/
_output_shapes
:?????????55
*
ksize
*
paddingVALID*
strides
23
1TinyVGG-Extra-Conv-Dense/max_pooling2d_39/MaxPool?
8TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2DConv2D:TinyVGG-Extra-Conv-Dense/max_pooling2d_39/MaxPool:output:0@TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
2,
*TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_80/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????33
2)
'TinyVGG-Extra-Conv-Dense/conv2d_80/Relu?
8TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02:
8TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2DConv2D5TinyVGG-Extra-Conv-Dense/conv2d_80/Relu:activations:0@TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
*
paddingVALID*
strides
2+
)TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D?
9TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOpReadVariableOpBtinyvgg_extra_conv_dense_conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp?
*TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D:output:0ATinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
2,
*TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd?
'TinyVGG-Extra-Conv-Dense/conv2d_81/ReluRelu3TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????11
2)
'TinyVGG-Extra-Conv-Dense/conv2d_81/Relu?
1TinyVGG-Extra-Conv-Dense/max_pooling2d_40/MaxPoolMaxPool5TinyVGG-Extra-Conv-Dense/conv2d_81/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
23
1TinyVGG-Extra-Conv-Dense/max_pooling2d_40/MaxPool?
)TinyVGG-Extra-Conv-Dense/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2+
)TinyVGG-Extra-Conv-Dense/flatten_16/Const?
+TinyVGG-Extra-Conv-Dense/flatten_16/ReshapeReshape:TinyVGG-Extra-Conv-Dense/max_pooling2d_40/MaxPool:output:02TinyVGG-Extra-Conv-Dense/flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????-2-
+TinyVGG-Extra-Conv-Dense/flatten_16/Reshape?
7TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOpReadVariableOp@tinyvgg_extra_conv_dense_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype029
7TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp?
(TinyVGG-Extra-Conv-Dense/dense_20/MatMulMatMul4TinyVGG-Extra-Conv-Dense/flatten_16/Reshape:output:0?TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(TinyVGG-Extra-Conv-Dense/dense_20/MatMul?
8TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/dense_20/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/dense_20/MatMul:product:0@TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd?
&TinyVGG-Extra-Conv-Dense/dense_20/ReluRelu2TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&TinyVGG-Extra-Conv-Dense/dense_20/Relu?
7TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOpReadVariableOp@tinyvgg_extra_conv_dense_dense_21_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype029
7TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp?
(TinyVGG-Extra-Conv-Dense/dense_21/MatMulMatMul4TinyVGG-Extra-Conv-Dense/dense_20/Relu:activations:0?TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2*
(TinyVGG-Extra-Conv-Dense/dense_21/MatMul?
8TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOpReadVariableOpAtinyvgg_extra_conv_dense_dense_21_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02:
8TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp?
)TinyVGG-Extra-Conv-Dense/dense_21/BiasAddBiasAdd2TinyVGG-Extra-Conv-Dense/dense_21/MatMul:product:0@TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2+
)TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd?
)TinyVGG-Extra-Conv-Dense/dense_21/SoftmaxSoftmax2TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2+
)TinyVGG-Extra-Conv-Dense/dense_21/Softmax?
IdentityIdentity3TinyVGG-Extra-Conv-Dense/dense_21/Softmax:softmax:0:^TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp:^TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp:^TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp:^TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp:^TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp:^TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp8^TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp9^TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp8^TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2v
9TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_76/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_76/Conv2D/ReadVariableOp2v
9TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_77/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_77/Conv2D/ReadVariableOp2v
9TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_78/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_78/Conv2D/ReadVariableOp2v
9TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_79/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_79/Conv2D/ReadVariableOp2v
9TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_80/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_80/Conv2D/ReadVariableOp2v
9TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp9TinyVGG-Extra-Conv-Dense/conv2d_81/BiasAdd/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp8TinyVGG-Extra-Conv-Dense/conv2d_81/Conv2D/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp8TinyVGG-Extra-Conv-Dense/dense_20/BiasAdd/ReadVariableOp2r
7TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp7TinyVGG-Extra-Conv-Dense/dense_20/MatMul/ReadVariableOp2t
8TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp8TinyVGG-Extra-Conv-Dense/dense_21/BiasAdd/ReadVariableOp2r
7TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp7TinyVGG-Extra-Conv-Dense/dense_21/MatMul/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
?	
?
D__inference_dense_20_layer_call_and_return_conditional_losses_104012

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????-
 
_user_specified_nameinputs
?

*__inference_conv2d_78_layer_call_fn_103930

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_78_layer_call_and_return_conditional_losses_1032322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????ll
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????nn
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????nn

 
_user_specified_nameinputs
?
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_103996

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????-2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????-2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?W
?

T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103796

inputs,
(conv2d_76_conv2d_readvariableop_resource-
)conv2d_76_biasadd_readvariableop_resource,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource,
(conv2d_78_conv2d_readvariableop_resource-
)conv2d_78_biasadd_readvariableop_resource,
(conv2d_79_conv2d_readvariableop_resource-
)conv2d_79_biasadd_readvariableop_resource,
(conv2d_80_conv2d_readvariableop_resource-
)conv2d_80_biasadd_readvariableop_resource,
(conv2d_81_conv2d_readvariableop_resource-
)conv2d_81_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource
identity?? conv2d_76/BiasAdd/ReadVariableOp?conv2d_76/Conv2D/ReadVariableOp? conv2d_77/BiasAdd/ReadVariableOp?conv2d_77/Conv2D/ReadVariableOp? conv2d_78/BiasAdd/ReadVariableOp?conv2d_78/Conv2D/ReadVariableOp? conv2d_79/BiasAdd/ReadVariableOp?conv2d_79/Conv2D/ReadVariableOp? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02!
conv2d_76/Conv2D/ReadVariableOp?
conv2d_76/Conv2DConv2Dinputs'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
conv2d_76/Conv2D?
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_76/BiasAdd/ReadVariableOp?
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2
conv2d_76/BiasAdd?
conv2d_76/ReluReluconv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
conv2d_76/Relu?
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_77/Conv2D/ReadVariableOp?
conv2d_77/Conv2DConv2Dconv2d_76/Relu:activations:0'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
conv2d_77/Conv2D?
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp?
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2
conv2d_77/BiasAdd?
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
conv2d_77/Relu?
max_pooling2d_38/MaxPoolMaxPoolconv2d_77/Relu:activations:0*/
_output_shapes
:?????????nn
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool?
conv2d_78/Conv2D/ReadVariableOpReadVariableOp(conv2d_78_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_78/Conv2D/ReadVariableOp?
conv2d_78/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_78/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
*
paddingVALID*
strides
2
conv2d_78/Conv2D?
 conv2d_78/BiasAdd/ReadVariableOpReadVariableOp)conv2d_78_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_78/BiasAdd/ReadVariableOp?
conv2d_78/BiasAddBiasAddconv2d_78/Conv2D:output:0(conv2d_78/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
2
conv2d_78/BiasAdd~
conv2d_78/ReluReluconv2d_78/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll
2
conv2d_78/Relu?
conv2d_79/Conv2D/ReadVariableOpReadVariableOp(conv2d_79_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_79/Conv2D/ReadVariableOp?
conv2d_79/Conv2DConv2Dconv2d_78/Relu:activations:0'conv2d_79/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
*
paddingVALID*
strides
2
conv2d_79/Conv2D?
 conv2d_79/BiasAdd/ReadVariableOpReadVariableOp)conv2d_79_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_79/BiasAdd/ReadVariableOp?
conv2d_79/BiasAddBiasAddconv2d_79/Conv2D:output:0(conv2d_79/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
2
conv2d_79/BiasAdd~
conv2d_79/ReluReluconv2d_79/BiasAdd:output:0*
T0*/
_output_shapes
:?????????jj
2
conv2d_79/Relu?
max_pooling2d_39/MaxPoolMaxPoolconv2d_79/Relu:activations:0*/
_output_shapes
:?????????55
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_80/Conv2D/ReadVariableOp?
conv2d_80/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
*
paddingVALID*
strides
2
conv2d_80/Conv2D?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_80/BiasAdd/ReadVariableOp?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????33
2
conv2d_80/BiasAdd~
conv2d_80/ReluReluconv2d_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????33
2
conv2d_80/Relu?
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02!
conv2d_81/Conv2D/ReadVariableOp?
conv2d_81/Conv2DConv2Dconv2d_80/Relu:activations:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
*
paddingVALID*
strides
2
conv2d_81/Conv2D?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 conv2d_81/BiasAdd/ReadVariableOp?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
2
conv2d_81/BiasAdd~
conv2d_81/ReluReluconv2d_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????11
2
conv2d_81/Relu?
max_pooling2d_40/MaxPoolMaxPoolconv2d_81/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPoolu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_16/Const?
flatten_16/ReshapeReshape!max_pooling2d_40/MaxPool:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????-2
flatten_16/Reshape?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?-?*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMulflatten_16/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_20/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_21/BiasAdd|
dense_21/SoftmaxSoftmaxdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_21/Softmax?
IdentityIdentitydense_21/Softmax:softmax:0!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOp!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp!^conv2d_78/BiasAdd/ReadVariableOp ^conv2d_78/Conv2D/ReadVariableOp!^conv2d_79/BiasAdd/ReadVariableOp ^conv2d_79/Conv2D/ReadVariableOp!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2D
 conv2d_78/BiasAdd/ReadVariableOp conv2d_78/BiasAdd/ReadVariableOp2B
conv2d_78/Conv2D/ReadVariableOpconv2d_78/Conv2D/ReadVariableOp2D
 conv2d_79/BiasAdd/ReadVariableOp conv2d_79/BiasAdd/ReadVariableOp2B
conv2d_79/Conv2D/ReadVariableOpconv2d_79/Conv2D/ReadVariableOp2D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_78_layer_call_and_return_conditional_losses_103232

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ll
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????ll
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????nn
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn

 
_user_specified_nameinputs
?

*__inference_conv2d_79_layer_call_fn_103950

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_79_layer_call_and_return_conditional_losses_1032592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????jj
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????ll
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????ll

 
_user_specified_nameinputs
?

*__inference_conv2d_80_layer_call_fn_103970

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????33
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_80_layer_call_and_return_conditional_losses_1032872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????33
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????55
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????55

 
_user_specified_nameinputs
?

?
E__inference_conv2d_79_layer_call_and_return_conditional_losses_103941

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????jj
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????jj
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????jj
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????ll
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????ll

 
_user_specified_nameinputs
?	
?
D__inference_dense_21_layer_call_and_return_conditional_losses_104032

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_76_layer_call_and_return_conditional_losses_103177

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103619
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_1035842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
input_17
?

?
E__inference_conv2d_78_layer_call_and_return_conditional_losses_103921

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ll
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????ll
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????nn
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn

 
_user_specified_nameinputs
?

?
E__inference_conv2d_81_layer_call_and_return_conditional_losses_103981

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????11
2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????11
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????11
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????33
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????33

 
_user_specified_nameinputs
?8
?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103499

inputs
conv2d_76_103454
conv2d_76_103456
conv2d_77_103459
conv2d_77_103461
conv2d_78_103465
conv2d_78_103467
conv2d_79_103470
conv2d_79_103472
conv2d_80_103476
conv2d_80_103478
conv2d_81_103481
conv2d_81_103483
dense_20_103488
dense_20_103490
dense_21_103493
dense_21_103495
identity??!conv2d_76/StatefulPartitionedCall?!conv2d_77/StatefulPartitionedCall?!conv2d_78/StatefulPartitionedCall?!conv2d_79/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_76_103454conv2d_76_103456*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_76_layer_call_and_return_conditional_losses_1031772#
!conv2d_76/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0conv2d_77_103459conv2d_77_103461*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_77_layer_call_and_return_conditional_losses_1032042#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_1031322"
 max_pooling2d_38/PartitionedCall?
!conv2d_78/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_78_103465conv2d_78_103467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_78_layer_call_and_return_conditional_losses_1032322#
!conv2d_78/StatefulPartitionedCall?
!conv2d_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_78/StatefulPartitionedCall:output:0conv2d_79_103470conv2d_79_103472*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_79_layer_call_and_return_conditional_losses_1032592#
!conv2d_79/StatefulPartitionedCall?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_1031442"
 max_pooling2d_39/PartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_80_103476conv2d_80_103478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????33
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_80_layer_call_and_return_conditional_losses_1032872#
!conv2d_80/StatefulPartitionedCall?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0conv2d_81_103481conv2d_81_103483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????11
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_81_layer_call_and_return_conditional_losses_1033142#
!conv2d_81/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_1031562"
 max_pooling2d_40/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)max_pooling2d_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????-* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_1033372
flatten_16/PartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_20_103488dense_20_103490*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1033562"
 dense_20/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_103493dense_21_103495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_1033832"
 dense_21/StatefulPartitionedCall?
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall"^conv2d_78/StatefulPartitionedCall"^conv2d_79/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:???????????::::::::::::::::2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!conv2d_78/StatefulPartitionedCall!conv2d_78/StatefulPartitionedCall2F
!conv2d_79/StatefulPartitionedCall!conv2d_79/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_104428
file_prefix%
!assignvariableop_conv2d_76_kernel%
!assignvariableop_1_conv2d_76_bias'
#assignvariableop_2_conv2d_77_kernel%
!assignvariableop_3_conv2d_77_bias'
#assignvariableop_4_conv2d_78_kernel%
!assignvariableop_5_conv2d_78_bias'
#assignvariableop_6_conv2d_79_kernel%
!assignvariableop_7_conv2d_79_bias'
#assignvariableop_8_conv2d_80_kernel%
!assignvariableop_9_conv2d_80_bias(
$assignvariableop_10_conv2d_81_kernel&
"assignvariableop_11_conv2d_81_bias'
#assignvariableop_12_dense_20_kernel%
!assignvariableop_13_dense_20_bias'
#assignvariableop_14_dense_21_kernel%
!assignvariableop_15_dense_21_bias!
assignvariableop_16_adam_iter#
assignvariableop_17_adam_beta_1#
assignvariableop_18_adam_beta_2"
assignvariableop_19_adam_decay*
&assignvariableop_20_adam_learning_rate
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1
assignvariableop_25_total_2
assignvariableop_26_count_2/
+assignvariableop_27_adam_conv2d_76_kernel_m-
)assignvariableop_28_adam_conv2d_76_bias_m/
+assignvariableop_29_adam_conv2d_77_kernel_m-
)assignvariableop_30_adam_conv2d_77_bias_m/
+assignvariableop_31_adam_conv2d_78_kernel_m-
)assignvariableop_32_adam_conv2d_78_bias_m/
+assignvariableop_33_adam_conv2d_79_kernel_m-
)assignvariableop_34_adam_conv2d_79_bias_m/
+assignvariableop_35_adam_conv2d_80_kernel_m-
)assignvariableop_36_adam_conv2d_80_bias_m/
+assignvariableop_37_adam_conv2d_81_kernel_m-
)assignvariableop_38_adam_conv2d_81_bias_m.
*assignvariableop_39_adam_dense_20_kernel_m,
(assignvariableop_40_adam_dense_20_bias_m.
*assignvariableop_41_adam_dense_21_kernel_m,
(assignvariableop_42_adam_dense_21_bias_m/
+assignvariableop_43_adam_conv2d_76_kernel_v-
)assignvariableop_44_adam_conv2d_76_bias_v/
+assignvariableop_45_adam_conv2d_77_kernel_v-
)assignvariableop_46_adam_conv2d_77_bias_v/
+assignvariableop_47_adam_conv2d_78_kernel_v-
)assignvariableop_48_adam_conv2d_78_bias_v/
+assignvariableop_49_adam_conv2d_79_kernel_v-
)assignvariableop_50_adam_conv2d_79_bias_v/
+assignvariableop_51_adam_conv2d_80_kernel_v-
)assignvariableop_52_adam_conv2d_80_bias_v/
+assignvariableop_53_adam_conv2d_81_kernel_v-
)assignvariableop_54_adam_conv2d_81_bias_v.
*assignvariableop_55_adam_dense_20_kernel_v,
(assignvariableop_56_adam_dense_20_bias_v.
*assignvariableop_57_adam_dense_21_kernel_v,
(assignvariableop_58_adam_dense_21_bias_v
identity_60??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*? 
value? B? <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_76_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_76_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_77_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_77_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_78_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_78_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_79_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_79_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_80_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_80_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_81_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_81_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_21_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_21_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_76_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_76_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_77_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_77_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_78_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_78_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_79_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_79_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_80_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_80_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_81_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_81_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_20_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_20_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_21_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_21_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_76_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_76_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_77_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_77_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_78_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_78_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_79_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_79_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_80_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_80_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_81_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_81_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_20_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_20_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_21_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_21_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59?

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

*__inference_conv2d_76_layer_call_fn_103890

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_76_layer_call_and_return_conditional_losses_1031772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_103156

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
input_17;
serving_default_input_17:0???????????<
dense_210
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?m
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?h
_tf_keras_sequential?h{"class_name": "Sequential", "name": "TinyVGG-Extra-Conv-Dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "TinyVGG-Extra-Conv-Dense", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_78", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_79", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_80", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "TinyVGG-Extra-Conv-Dense", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_78", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_79", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_80", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "f1", "dtype": "float32", "fn": "f1"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_76", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_77", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 222, 222, 10]}}
?
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_78", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_78", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 110, 110, 10]}}
?	

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_79", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 108, 108, 10]}}
?
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_80", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 53, 53, 10]}}
?	

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_81", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 51, 10]}}
?
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5760}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5760]}}
?

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratem?m?m?m?#m?$m?)m?*m?3m?4m?9m?:m?Gm?Hm?Mm?Nm?v?v?v?v?#v?$v?)v?*v?3v?4v?9v?:v?Gv?Hv?Mv?Nv?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15"
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15"
trackable_list_wrapper
?
regularization_losses
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_regularization_losses
	variables
trainable_variables
\layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(
2conv2d_76/kernel
:
2conv2d_76/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
]non_trainable_variables
^metrics
_layer_regularization_losses

`layers
	variables
trainable_variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(

2conv2d_77/kernel
:
2conv2d_77/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
bnon_trainable_variables
cmetrics
dlayer_regularization_losses

elayers
	variables
trainable_variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
gnon_trainable_variables
hmetrics
ilayer_regularization_losses

jlayers
 	variables
!trainable_variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(

2conv2d_78/kernel
:
2conv2d_78/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
%regularization_losses
lnon_trainable_variables
mmetrics
nlayer_regularization_losses

olayers
&	variables
'trainable_variables
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(

2conv2d_79/kernel
:
2conv2d_79/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
+regularization_losses
qnon_trainable_variables
rmetrics
slayer_regularization_losses

tlayers
,	variables
-trainable_variables
ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/regularization_losses
vnon_trainable_variables
wmetrics
xlayer_regularization_losses

ylayers
0	variables
1trainable_variables
zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(

2conv2d_80/kernel
:
2conv2d_80/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5regularization_losses
{non_trainable_variables
|metrics
}layer_regularization_losses

~layers
6	variables
7trainable_variables
layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(

2conv2d_81/kernel
:
2conv2d_81/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
;regularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
<	variables
=trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
@	variables
Atrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
D	variables
Etrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?-?2dense_20/kernel
:?2dense_20/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
Iregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
J	variables
Ktrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?
2dense_21/kernel
:
2dense_21/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
Oregularization_losses
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layers
P	variables
Qtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "f1", "dtype": "float32", "config": {"name": "f1", "dtype": "float32", "fn": "f1"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-
2Adam/conv2d_76/kernel/m
!:
2Adam/conv2d_76/bias/m
/:-

2Adam/conv2d_77/kernel/m
!:
2Adam/conv2d_77/bias/m
/:-

2Adam/conv2d_78/kernel/m
!:
2Adam/conv2d_78/bias/m
/:-

2Adam/conv2d_79/kernel/m
!:
2Adam/conv2d_79/bias/m
/:-

2Adam/conv2d_80/kernel/m
!:
2Adam/conv2d_80/bias/m
/:-

2Adam/conv2d_81/kernel/m
!:
2Adam/conv2d_81/bias/m
(:&
?-?2Adam/dense_20/kernel/m
!:?2Adam/dense_20/bias/m
':%	?
2Adam/dense_21/kernel/m
 :
2Adam/dense_21/bias/m
/:-
2Adam/conv2d_76/kernel/v
!:
2Adam/conv2d_76/bias/v
/:-

2Adam/conv2d_77/kernel/v
!:
2Adam/conv2d_77/bias/v
/:-

2Adam/conv2d_78/kernel/v
!:
2Adam/conv2d_78/bias/v
/:-

2Adam/conv2d_79/kernel/v
!:
2Adam/conv2d_79/bias/v
/:-

2Adam/conv2d_80/kernel/v
!:
2Adam/conv2d_80/bias/v
/:-

2Adam/conv2d_81/kernel/v
!:
2Adam/conv2d_81/bias/v
(:&
?-?2Adam/dense_20/kernel/v
!:?2Adam/dense_20/bias/v
':%	?
2Adam/dense_21/kernel/v
 :
2Adam/dense_21/bias/v
?2?
!__inference__wrapped_model_103126?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
input_17???????????
?2?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103448
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103400
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103796
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103731?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103534
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103833
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103619
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103870?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_76_layer_call_and_return_conditional_losses_103881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_76_layer_call_fn_103890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_77_layer_call_and_return_conditional_losses_103901?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_77_layer_call_fn_103910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_103132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_38_layer_call_fn_103138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2d_78_layer_call_and_return_conditional_losses_103921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_78_layer_call_fn_103930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_79_layer_call_and_return_conditional_losses_103941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_79_layer_call_fn_103950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_103144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_39_layer_call_fn_103150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2d_80_layer_call_and_return_conditional_losses_103961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_80_layer_call_fn_103970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_81_layer_call_and_return_conditional_losses_103981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_81_layer_call_fn_103990?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_103156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_40_layer_call_fn_103162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_flatten_16_layer_call_and_return_conditional_losses_103996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_16_layer_call_fn_104001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_20_layer_call_and_return_conditional_losses_104012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_20_layer_call_fn_104021?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_21_layer_call_and_return_conditional_losses_104032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_21_layer_call_fn_104041?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_103666input_17"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103400~#$)*349:GHMNC?@
9?6
,?)
input_17???????????
p

 
? "%?"
?
0?????????

? ?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103448~#$)*349:GHMNC?@
9?6
,?)
input_17???????????
p 

 
? "%?"
?
0?????????

? ?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103731|#$)*349:GHMNA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????

? ?
T__inference_TinyVGG-Extra-Conv-Dense_layer_call_and_return_conditional_losses_103796|#$)*349:GHMNA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????

? ?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103534q#$)*349:GHMNC?@
9?6
,?)
input_17???????????
p

 
? "??????????
?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103619q#$)*349:GHMNC?@
9?6
,?)
input_17???????????
p 

 
? "??????????
?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103833o#$)*349:GHMNA?>
7?4
*?'
inputs???????????
p

 
? "??????????
?
9__inference_TinyVGG-Extra-Conv-Dense_layer_call_fn_103870o#$)*349:GHMNA?>
7?4
*?'
inputs???????????
p 

 
? "??????????
?
!__inference__wrapped_model_103126?#$)*349:GHMN;?8
1?.
,?)
input_17???????????
? "3?0
.
dense_21"?
dense_21?????????
?
E__inference_conv2d_76_layer_call_and_return_conditional_losses_103881p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????

? ?
*__inference_conv2d_76_layer_call_fn_103890c9?6
/?,
*?'
inputs???????????
? ""????????????
?
E__inference_conv2d_77_layer_call_and_return_conditional_losses_103901p9?6
/?,
*?'
inputs???????????

? "/?,
%?"
0???????????

? ?
*__inference_conv2d_77_layer_call_fn_103910c9?6
/?,
*?'
inputs???????????

? ""????????????
?
E__inference_conv2d_78_layer_call_and_return_conditional_losses_103921l#$7?4
-?*
(?%
inputs?????????nn

? "-?*
#? 
0?????????ll

? ?
*__inference_conv2d_78_layer_call_fn_103930_#$7?4
-?*
(?%
inputs?????????nn

? " ??????????ll
?
E__inference_conv2d_79_layer_call_and_return_conditional_losses_103941l)*7?4
-?*
(?%
inputs?????????ll

? "-?*
#? 
0?????????jj

? ?
*__inference_conv2d_79_layer_call_fn_103950_)*7?4
-?*
(?%
inputs?????????ll

? " ??????????jj
?
E__inference_conv2d_80_layer_call_and_return_conditional_losses_103961l347?4
-?*
(?%
inputs?????????55

? "-?*
#? 
0?????????33

? ?
*__inference_conv2d_80_layer_call_fn_103970_347?4
-?*
(?%
inputs?????????55

? " ??????????33
?
E__inference_conv2d_81_layer_call_and_return_conditional_losses_103981l9:7?4
-?*
(?%
inputs?????????33

? "-?*
#? 
0?????????11

? ?
*__inference_conv2d_81_layer_call_fn_103990_9:7?4
-?*
(?%
inputs?????????33

? " ??????????11
?
D__inference_dense_20_layer_call_and_return_conditional_losses_104012^GH0?-
&?#
!?
inputs??????????-
? "&?#
?
0??????????
? ~
)__inference_dense_20_layer_call_fn_104021QGH0?-
&?#
!?
inputs??????????-
? "????????????
D__inference_dense_21_layer_call_and_return_conditional_losses_104032]MN0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? }
)__inference_dense_21_layer_call_fn_104041PMN0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_flatten_16_layer_call_and_return_conditional_losses_103996a7?4
-?*
(?%
inputs?????????

? "&?#
?
0??????????-
? ?
+__inference_flatten_16_layer_call_fn_104001T7?4
-?*
(?%
inputs?????????

? "???????????-?
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_103132?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_38_layer_call_fn_103138?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_103144?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_39_layer_call_fn_103150?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_103156?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_40_layer_call_fn_103162?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_103666?#$)*349:GHMNG?D
? 
=?:
8
input_17,?)
input_17???????????"3?0
.
dense_21"?
dense_21?????????
