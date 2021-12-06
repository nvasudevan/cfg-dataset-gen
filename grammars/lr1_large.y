%define lr.type canonical-lr

%start root

%%

root: K 'm' H | V 'c' 'z' | B H 'z' | C 't' 'm'
;
A: K 'u' 'c' | L 'a' V | K 'z' 'c' | 'a' V 'a' V | G H
;
B: S | 'h' 'k' | 'z' 'r' 'k' 'l' | 'a' C 'c' M | R
;
C: 'q' 'r' 'n' | 'p' 'b' 'h' | 'k' 'b' 'n'
;
G: 'f' 't' | M | 'm' 'n' | 'b' M
;
H: 'n' 'o' 't' | M | 'l' 'j' R | S | 'f' 'i' 'l' | 'l' 'o' 't'
;
J: 'z' | 'r' 'i' | 'r' | 'z' 'i' | 'r' 'm' | 'l' 'l' | 'l'
;
K: 'a' 'z' | 'b' 'z' | G 'o' N | J N
;
L: 'u' V | 'z' V | 'r' H | 'r' 'm' M | M 'r' 'o' B
;
M: 'n' | 'v' | 'i' | 'o' R | 'j' C
;
N: C | 'g' 'b' | 'z' 'b' | 'b' 'b' | 'g' 'o' | 'z' 'o' | 'k' 'o'
;
Q: V 'c' | 's' | 't' | L 'r' A | 'b' | 'g' | G H S
;
S: 's' 't' | 'j' 'i' | 'v' 'u'
;
R: 'f' 'f' C | 'z' 'f' | 'g' 'f' | 'c' 'c' 'r' | 'o' N
;
U: 'c' 'a' | Q | 'r' 'a' | 'a' H N 'l'
;
V: 'r' 'c' 'k' | 'c' K U | M | 'c' 'm' 'z' 'c' | 'a' 'i'
;
%%
