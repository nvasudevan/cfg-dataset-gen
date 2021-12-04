%define lr.type canonical-lr

%start root

%%

root: K 'm' H | V 'c' 'z' | B H 'z'
;
B: 'h' 'k' | 'z' 'r' 'k' 'l' | 'a' 'c' M
;
K: 'a' 'z' | 'b' 'z' | G 'o' N | J N
;
V: 'r' | 'c' K U | M | 'c' 'm' 'z' 'c' 'a'
;
A: K 'u' 'c' | L 'a' V | K 'z' 'c' | 'a' V 'a' V | G H
;
Q: V 'c' | 's' | L 'r' A | 'b' | 'g' | G H
;
U: 'c' 'a' | Q | 'r' 'a' | 'a' H N 'l'
;
L: 'u' V | 'z' V | 'r' H | 'r' 'm' M | M 'r' 'o' B
;
J: 'z' | 'r' 'i' | 'r' | 'z' 'i' | 'r' 'm' | 'l' 'l' | 'l'
;
N: 'g' 'b' | 'z' 'b' | 'b' 'b' | 'g' 'o' | 'z' 'o'
;
M: 'n' | 'v' | 'i' | 'o'
;
G: M | 'm' 'n' | 'b' M
;
H: 'n' 'o' | M | 'l' 'j'
;
%%
