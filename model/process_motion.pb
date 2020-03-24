
>
inputPlaceholder*
dtype0*
shape:hN
&
norm/mulMulinputinput*
T0
H
norm/Sum/reduction_indicesConst*
valueB:*
dtype0
[
norm/SumSumnorm/mulnorm/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
$
	norm/SqrtSqrtnorm/Sum*
T0
x
AvgPoolAvgPool	norm/Sqrt*
ksize
hN*
paddingSAME*
T0*
data_formatNHWC*
strides
hN
$
outputIdentityAvgPool*
T0"w