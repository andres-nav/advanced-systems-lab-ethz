{
  description = "Development environment for the Advanced Systems Lab project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs = { self, systems, nixpkgs, ... }:
    let
      eachSystem = f: nixpkgs.lib.genAttrs (import systems) (system: f {
        pkgs = nixpkgs.legacyPackages.${system};
      });
    in {
      devShells = eachSystem ({ pkgs }:
        let
          python = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.matplotlib
          ]);
          texlive = pkgs.texlive.combine {
            inherit (pkgs.texlive) scheme-full;
          };
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.gnumake
              pkgs.gcc
              pkgs.clang-tools  # clangd, clang-format, clang-tidy

              python

              # LaTeX toolchain for report/report.tex
              texlive
              pkgs.texlab
              pkgs.tex-fmt
            ];
          };
        });
    };
}
