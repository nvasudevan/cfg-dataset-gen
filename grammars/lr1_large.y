%define lr.type canonical-lr

%start root

%%

root: K 'm' H | V 'c' 'z' | B H 'z' | 't' 'm'
;
R: 'f' 'f' | 'z' 'f' | 'g' 'f' | 'c' 'c' 'r' | 'o' N
;
B: S | 'h' 'k' | 'z' 'r' 'k' 'l' | 'a' 'c' M | R
;
K: 'a' 'z' | 'b' 'z' | G 'o' N | J N
;
V: 'r' | 'c' K U | M | 'c' 'm' 'z' 'c' 'a'
;
A: K 'u' 'c' | L 'a' V | K 'z' 'c' | 'a' V 'a' V | G H
;
Q: V 'c' | 's' | 't' | L 'r' A | 'b' | 'g' | G H S
;
U: 'c' 'a' | Q | 'r' 'a' | 'a' H N 'l'
;
L: 'u' V | 'z' V | 'r' H | 'r' 'm' M | M 'r' 'o' B
;
J: 'z' | 'r' 'i' | 'r' | 'z' 'i' | 'r' 'm' | 'l' 'l' | 'l'
;
N: 'g' 'b' | 'z' 'b' | 'b' 'b' | 'g' 'o' | 'z' 'o'
;
M: 'n' | 'v' | 'i' | 'o' R | 'j' R
;
G: 'f' 't' | M | 'm' 'n' | 'b' M
;
H: 'n' 'o' 't' | M | 'l' 'j' R | S
;
S: 's' 't' | 'j' 'i' | 'v' 'u' 
;
%%
