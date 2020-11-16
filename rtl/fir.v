module fir
#(
    parameter BW=12,
    parameter N = 5,
    parameter S = BW-1,
)
(
    input          CK,
    input          RB,
    input [BW-1:0] X,
    input [N*BW-1:0] C,
    output [BW-1:0] Y
);

    reg [2*BW+N:0] tap[1:N-1];
    wire    [2*BW:0] Yfull;
    wire    [2*BW:0] YfullShifted;



    always @(posedge CK or negedge RB)
        if (!RB)
            tap[N-1] <= {2*BW+N+1{1'b0}};
        else
            tap[N-1] <= $signed(C[N*BW-1:(N-1)*BW])*$signed(X);

    generate    
        genvar i;
        for (i=1; i<N-1; i=i+1)
            always @(posedge CK or negedge RB)
                if (!RB)
                    tap[i] <= {2*BW+N+1{1'b0}};
                else
                    tap[i]<=$signed(C[(i+1)*BW-1:i*BW])*$signed(X)+$signed(tap[i+1]); 
    endgenerate

    assign Yfull=$signed(tap[1])+$signed(X)*$signed(C[BW-1:0]);
    assign YfullShifted = ($signed(Yfull))>>>(S);
    //assign Y=Yfull[2*BW-2:BW-1]; //Closest to the gain of 1 for 
    assign Y=YfullShifted[BW-1:0];  

endmodule

