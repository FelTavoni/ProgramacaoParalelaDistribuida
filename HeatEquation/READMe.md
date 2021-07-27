# FDM in Heat Equation

> This repository implements a parallelized version of the FDM in Heat Equation proposed in the 14th Marathon of Parallel Programming presented by Universidade Mackenzie. More details about the code to be added...


### ⚙️ Adjustments and improvements

This project still under development and the next updates will be aiming the following tasks:

- [ ] CUDA returned answer is different from the sequential code. Check if it's a precision problem...
- [ ] A more detailed explanation of the CUDA code on READMe.

## 💻 Prerequisites

Before you start, check if you have matched the following requisites:
* Instal the following tools and their versions `< C \ CUDA >`.
* Windows as OS.

## 🚀 Compiling FDM in Heat Equation

To compile the program, simply run in command line:

Windows:
```
make
```

## ☕ Using FDM in Heat Equation

To use the project FDM in Heat Equation, simply run the executable:

- Windows:

  - Sequential code:

    ```
    mdf.exe
    ```

  - Cuda code:

    ```
    mdf_cuda.exe
    ```

A input test is available, the `fdm.in` file. It's possible to execute it by running the following command.

```
mdf.exe < fdm.in
```

The same for the cuda program.

## 🤝 Contributors

This project is presented by:

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/56005905?v=4" width="100px;" alt="Felipe Tavoni's profile pic on GitHub"/><br>
        <sub>
          <b>Felipe Tavoni</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

<!-- ## 📝 License

This project is under a licence. Check the file [LICENSE](LICENSE.md) for more details. -->