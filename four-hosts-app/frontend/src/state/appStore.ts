import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

type Role = "free" | "basic" | "pro" | "enterprise" | "admin";

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: {
    id: string;
    email: string;
    username: string;
    role: Role;
  } | null;
}

interface UIState {
  wsConnected: boolean;
  activeResearchId: string | null;
}

interface AppStore extends AuthState, UIState {
  setTokens: (access: string | null, refresh: string | null) => void;
  setUser: (user: AuthState["user"]) => void;
  setWsConnected: (v: boolean) => void;
  setActiveResearch: (id: string | null) => void;
  reset: () => void;
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      wsConnected: false,
      activeResearchId: null,

      setTokens: (access, refresh) => set({ accessToken: access, refreshToken: refresh }),
      setUser: (user) => set({ user }),
      setWsConnected: (v) => set({ wsConnected: v }),
      setActiveResearch: (id) => set({ activeResearchId: id }),
      reset: () =>
        set({
          accessToken: null,
          refreshToken: null,
          user: null,
          wsConnected: false,
          activeResearchId: null,
        }),
    }),
    {
      name: "app-store",
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        accessToken: s.accessToken,
        refreshToken: s.refreshToken,
        user: s.user,
        activeResearchId: s.activeResearchId,
      }),
    }
  )
);