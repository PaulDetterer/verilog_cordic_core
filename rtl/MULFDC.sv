// Angle representation is always a 0.bbbbbb of a circle
// 100000 -> half circle (pi)
// 111111 -> 2*pi (almost)


module MULFDC
#(
    parameter BW = 12,
    parameter ABW = 10
)
(
    input clk_fs,  //  Sampling clock
    input clk_fso2,  // Decimation clock fsample/2
    input clk_fso4,  // Decimation clock fsample/4
    input rstb, // Reset

    input [BW-1:0] Iin,
    input [BW-1:0] Qin,
    input [ABW-1:0] Wif,
    output [BW-1:0] Iout,
    output [BW-1:0] Qout
);
    reg  [BW-1:0] Iin_s, Qin_s;
    wire  [BW-1:0] Irt, Qrt;

    // Sample
    always_ff @(posedge clk_fs or negedge rstb)
        if (rstb == 0)
         begin
            Iin_s <= 'd0;
            Qin_s <= 'd0;
         end
        else
         begin
            Iin_s <= Iin;
            Qin_s <= Qin;
         end

     // Rotate
     shifter u_shifter(
         .i_in(Iin_s),
         .q_in(Qin_s),
         .rst_neg(rstb),
         .clk(clk_fs),
         .bypass(0),
         .i_out(Irt),
         .q_out(Qrt)
     );

    
    // Filter 1
    wire [BW-1:0] Ilpf1,Qlpf1;
    reg  [BW-1:0] Idec1,Qdec1;

    fir #(.BW(BW),.N(5),.S(BW-1)) uIFIR1 (
        .CK(clk_fs),
        .RB(rstb),
        .X(Irt),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Ilpf1));
    
    fir #(.BW(BW),.N(5),.S(BW-1)) uQFIR1 (
        .CK(clk_fs),
        .RB(rstb),
        .X(Qrt),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Qlpf1));

    //Decimate 1

    always_ff @(posedge clk_fso2 or negedge rstb)
        if (rstb==0)
         begin
            Idec1 <= 'd0;
            Qdec1 <= 'd0;
         end
        else
         begin
            Idec1 <= Ilpf1;
            Qdec1 <= Qlpf1;
         end

    //Filter 2

    
    wire [BW-1:0] Ilpf2,Qlpf2;
    reg  [BW-1:0] Idec2,Qdec2;

    fir #(.BW(BW),.N(5),.S(BW-1)) uIFIR2 (
        .CK(clk_fso2),
        .RB(rstb),
        .X(Idec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Ilpf2));
    
    fir #(.BW(BW),.N(5),.S(BW-1)) uQFIR2 (
        .CK(clk_fso2),
        .RB(rstb),
        .X(Qdec1),
        .C({12'd211,12'd420,12'd503,12'd420,12'd211}),
        .Y(Qlpf2));
    
    //Decimate 2


    always_ff @(posedge clk_fso4 or negedge rstb)
        if (rstb==0)
         begin
            Idec2 <= 'd0;
            Qdec2 <= 'd0;
         end
        else
         begin
            Idec2 <= Ilpf2;
            Qdec2 <= Qlpf2;
         end

    assign Qout = Qdec2;
    assign Iout = Idec2;

endmodule    
