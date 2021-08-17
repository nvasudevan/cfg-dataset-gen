%define lr.type canonical-lr

%start root

%%

root: K 'm' H | V 'c' 'z'
;
K: 'a' 'z' | 'b' 'z' | G 'o' N | J N
;
V: 'r' | 'c' K U | M | 'c' 'm' 'z' 'c' 'a'
;
A: K 'u' 'c' | L 'a' V | K 'z' 'c' | 'a' V 'a' V | G H
;
Q: V 'c' | 's' | L 'r' A | 'b' | 'g' | G H
;
U: 'c' 'a' | Q | 'r' 'a' | 'a' H N
;
L: 'u' V | 'z' V | 'r' H | 'r' 'm' M | M 'r' 'o'
;
J: 'z' | 'r' 'i' | 'r' | 'z' 'i' | 'r' 'm'
;
N: 'g' 'b' | 'z' 'b' | 'b' 'b' | 'g' 'o'
;
M: 'n' | 'v' | 'i' | 'o'
;
G: M | 'm' 'n'
;
H: 'n' 'o' | M
;
%%
