from manim import *

class HybridArchitecture(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # 1. TÍTULO Y CONFIGURACIÓN BASE
        # ---------------------------------------------------------
        title = Text("Arquitectura Híbrida: MPI + OpenMP", font_size=40, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # ---------------------------------------------------------
        # 2. EL DATASET GLOBAL (Capa Superior)
        # ---------------------------------------------------------
        dataset = Rectangle(width=6, height=1.2, color=BLUE, fill_opacity=0.2)
        dataset.set_stroke(width=3)
        ds_text = Text("Dataset Global (Memoria Principal)", font_size=24).move_to(dataset.get_center())
        
        ds_group = VGroup(dataset, ds_text).shift(UP * 2)
        self.play(FadeIn(ds_group, shift=DOWN))
        self.wait(0.5)

        # ---------------------------------------------------------
        # 3. LOS NODOS MPI (Capa Inferior)
        # ---------------------------------------------------------
        nodes = VGroup()
        for i in range(4):
            # Caja del servidor (Nodo)
            node_box = Rectangle(width=2.8, height=2.5, color=ORANGE, fill_opacity=0.1)
            node_label = Text(f"Nodo MPI {i}", font_size=20, color=ORANGE).next_to(node_box.get_top(), DOWN, buff=0.2)
            
            # Los 4 Hilos OpenMP (Círculos dentro del servidor)
            threads = VGroup()
            for j in range(4):
                thread = Circle(radius=0.2, color=GREEN, fill_opacity=0.5)
                threads.add(thread)
            
            # Ordenar hilos en una cuadrícula 2x2
            threads.arrange_in_grid(rows=2, cols=2, buff=0.4)
            threads.move_to(node_box.get_center()).shift(DOWN * 0.2)
            
            # Agrupar todo lo del nodo
            n_group = VGroup(node_box, node_label, threads)
            nodes.add(n_group)
            
        # Distribuir los 4 nodos en horizontal
        nodes.arrange(RIGHT, buff=0.5).shift(DOWN * 1.5)
        
        # ---------------------------------------------------------
        # FASE A: MPI SCATTER (Distribución)
        # ---------------------------------------------------------
        step_text = Text("1. Distribución en Red (MPI_Scatter / MPI_Alltoallv)", font_size=28, color=BLUE)
        step_text.to_edge(DOWN)
        
        self.play(Write(step_text))
        
        arrows_down = VGroup()
        for i in range(4):
            arrow = Arrow(start=dataset.get_bottom(), end=nodes[i].get_top(), color=BLUE, buff=0.1)
            arrows_down.add(arrow)

        self.play(Create(arrows_down), run_time=1.5)
        self.play(FadeIn(nodes, shift=UP))
        self.wait(1)

        # ---------------------------------------------------------
        # FASE B: OPENMP (Cálculo Multihilo)
        # ---------------------------------------------------------
        step_text_2 = Text("2. Cálculo Multihilo Concurrente (#pragma omp parallel)", font_size=28, color=GREEN)
        step_text_2.to_edge(DOWN)
        
        self.play(Transform(step_text, step_text_2))
        self.play(FadeOut(arrows_down))
        
        # Animación de cómputo (parpadeo de los hilos)
        flash_animations = []
        for n_group in nodes:
            threads = n_group[2]
            for thread in threads:
                flash_animations.append(Indicate(thread, color=YELLOW, scale_factor=1.5))
        
        # Ejecutar el parpadeo un par de veces para simular la CPU al 100%
        self.play(*flash_animations, run_time=1.5)
        self.play(*flash_animations, run_time=1.5)
        self.wait(1)

        # ---------------------------------------------------------
        # FASE C: MPI ALLREDUCE (Fusión Global)
        # ---------------------------------------------------------
        step_text_3 = Text("3. Sincronización Global (MPI_Allreduce)", font_size=28, color=RED)
        step_text_3.to_edge(DOWN)
        
        self.play(Transform(step_text, step_text_3))
        
        arrows_up = VGroup()
        for i in range(4):
            arrow = Arrow(start=nodes[i].get_top(), end=dataset.get_bottom(), color=RED, buff=0.1)
            arrows_up.add(arrow)

        self.play(Create(arrows_up), run_time=1.5)
        
        # El dataset recibe los datos y se colorea de verde (completado)
        self.play(dataset.animate.set_color(GREEN).set_fill(GREEN, opacity=0.4))
        self.wait(2)
        
        # ---------------------------------------------------------
        # DESPEDIDA
        # ---------------------------------------------------------
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)