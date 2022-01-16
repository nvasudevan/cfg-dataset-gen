%define lr.type canonical-lr
%start root

%token TK_PQR TK_XYZ TK_MNP
%%

root: pqr | id_xyz
;

pqr: TK_PQR '/' TK_XYZ | TK_PQR '+' TK_XYZ | TK_XYZ '-' TK_XYZ | TK_PQR '*' TK_PQR |
;

id_xyz: TK_MNP '%' TK_MNP | TK_MNP '=' TK_PQR
;

%%